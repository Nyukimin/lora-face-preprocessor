# split_crop_faces.py
# ------------------------------------------------------------
# 目的:
#   - input/ に混在している画像（全身〜顔寄り）を自動仕分けし
#   - 顔検出できたものは 1024x1024 の顔中心クロップを作る
#   - false positive（顔じゃない誤検出）を減らすために
#       1) frontal + profile を拾う
#       2) eye 検証で弾く
#       3) 顔占有率（face_area/img_area）で小さすぎる検出を弾く
#
# 出力:
#   out/
#     A_fullbody/   : 顔が小さい（または検出できない）想定の画像
#     B_upper/      : 上半身想定
#     C_closeup/    : 顔寄り想定
#     cropped_1024/ : 1024x1024 にクロップした学習候補
#     report.csv    : 判定ログ
#
# メモ:
#   - Haar cascade は軽いが誤検出しやすい。そこで「検出後に弾く」が安定。
#   - 「髪型を学習させたくない」なら CROP_SCALE を小さめにし、
#     さらに CROP_ONLY_FOR_BC を True にして A_fullbody を学習から外す運用が安全。
# ------------------------------------------------------------

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# パラメータ（ここだけ触れば挙動を調整できる）
# ============================================================

# 入力フォルダ（この下に画像を置く）
INPUT_DIR = Path("input")

# 出力フォルダ（毎回削除して作り直す）
OUT_DIR = Path("out")

# --- 仕分け基準（face_area / img_area） ---
# 顔が画像のどれくらいの面積かで A/B/C に分類する
# 例:
#   face_ratio < A_MAX        -> A_fullbody（顔がかなり小さい）
#   A_MAX <= face_ratio < B_MAX -> B_upper（上半身）
#   B_MAX <= face_ratio       -> C_closeup（顔寄り）
A_MAX = 0.05
B_MAX = 0.15

# --- 誤検出対策: 顔占有率が小さすぎる検出は捨てる ---
# 背景や小物などを「顔」と誤検出すると、face_ratio が小さいことが多い
# ただし「全身で顔が小さい写真」もここで捨てられる（仕様）
# 目安:
#   0.05: 全身も少し拾う（誤検出増えやすい）
#   0.08: バランス（推奨）
#   0.12: 厳しめ（上半身〜顔寄り中心）
MIN_FACE_AREA_RATIO = 0.05

# --- クロップの寄り具合 ---
# 顔bboxを中心に正方形クロップを作るとき、bboxを何倍に広げるか
# 小さいほど「髪・背景」が入りにくい（髪型を学習させたくないなら小さめ）
# 目安:
#   1.4〜1.6: 髪を抑えたい（推奨）
#   1.7〜2.0: 余白多め（雰囲気も含めたいとき）
CROP_SCALE = 1.6

# SDXL 学習向けの出力サイズ（基本は 1024）
OUTPUT_SIZE = 1024

# Falseだと、クロップ結果が 1024 未満の場合は拡大しない（画質の崩れ防止）
# Trueだと、常に1024に揃える（学習用としては扱いやすい）
ALLOW_UPSCALE = False

# --- OpenCV検出パラメータ ---
# MIN_FACE_SIZE:
#   小さすぎると誤検出が増える / 大きすぎると全身写真を取りこぼす
#   全身多いなら 30〜50、顔寄り中心なら 80〜140
MIN_FACE_SIZE = 30

# 目検出の最小サイズ
EYE_MIN_SIZE = 18

# minNeighbors:
#   大きいほど厳しく（誤検出減るが取りこぼす）
#   小さいほど甘く（検出率上がるが誤検出増える）
MIN_NEIGHBORS_FACE = 3
MIN_NEIGHBORS_EYE = 5

# --- 誤検出対策: 目で検証する ---
# Trueにすると「目が見つかる顔bboxだけを採用」する
# 事故（指・胸・背景など）を激減できるが、横顔や伏し目は落ちやすい
REQUIRE_EYES = True

# 目が何個見つかればOKか
# 2にすると正面寄りのみになりやすい。横顔も拾うなら 1。
MIN_EYES = 1

# 目が顔bboxの上側にいるはず、という制約（下の方の誤検出を弾く）
EYE_REGION_Y_MAX = 0.70

# Trueの場合、cropped_1024 を B/C のみ出す（A_fullbody は出さない）
# 「顔LoRA」で髪型や背景を学習したくないなら True 推奨
CROP_ONLY_FOR_BC = True


# ============================================================
# ユーティリティ関数
# ============================================================

def ensure_dirs() -> None:
    """出力フォルダ配下を作る（存在してもOK）"""
    (OUT_DIR / "A_fullbody").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "B_upper").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "C_closeup").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cropped_1024").mkdir(parents=True, exist_ok=True)


def reset_output_dir():
    """
    毎回 out/ を削除してクリーンな状態で再生成する。
    - 過去の結果が混ざらない
    - 設定変更の結果が比較しやすい
    """
    if OUT_DIR.exists():
        print(f"[INFO] Removing existing output dir: {OUT_DIR.resolve()}")
        shutil.rmtree(OUT_DIR)


def pil_open_rgb(path: Path) -> Image.Image:
    """PILで画像を読み、RGBへ揃える（pngのRGBAなどを吸収）"""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def clamp(v: int, lo: int, hi: int) -> int:
    """範囲に収める（クロップが画像外に出ないようにする）"""
    return max(lo, min(hi, v))


def imread_bgr(path: Path):
    """
    OpenCVのimreadはWindowsの日本語パスで失敗することがあるので、
    np.fromfile + imdecode で読む（安定）
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_cascade_xml(name: str) -> str | None:
    """
    OpenCV付属のxml(haarcascade_*.xml)の場所を探す。
    - cv2.data が無い環境でも動くように、自前探索にしている
    """
    base = Path(cv2.__file__).resolve().parent

    # よくある配置候補
    candidates = [
        base / "data" / name,
        base.parent / "data" / name,
        base / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # 最後の手段: package配下を探索
    try:
        for p in base.rglob(name):
            return str(p)
    except Exception:
        pass

    return None


def load_cascade(name: str) -> cv2.CascadeClassifier:
    """
    Haar cascade xmlを読み込み、CascadeClassifierを返す。
    読めなかったら例外にする（黙って失敗すると原因が追いにくい）
    """
    p = find_cascade_xml(name)
    if not p:
        raise FileNotFoundError(f"Could not find cascade xml: {name}")
    c = cv2.CascadeClassifier(p)
    if c.empty():
        raise RuntimeError(f"Failed to load cascade: {p}")
    return c


# ============================================================
# Cascade 読み込み（ここが通れば OpenCV環境はOK）
# ============================================================

# 正面顔
FACE_FRONTAL = load_cascade("haarcascade_frontalface_default.xml")

# 横顔（左右向き）
FACE_PROFILE = load_cascade("haarcascade_profileface.xml")

# 目（誤検出除去に使う）
EYE_CASCADE = load_cascade("haarcascade_eye.xml")


# ============================================================
# 顔検出・候補選択
# ============================================================

def detect_faces(gray: np.ndarray):
    """
    正面＋横顔（＋横顔を左右反転したもの）で候補bboxを集める。
    戻り値: [(x,y,w,h,kind), ...]
    """
    faces = []

    # 正面
    f1 = FACE_FRONTAL.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=MIN_NEIGHBORS_FACE,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    for (x, y, w, h) in (f1 if f1 is not None else []):
        faces.append((int(x), int(y), int(w), int(h), "frontal"))

    # 横顔（片方向）
    f2 = FACE_PROFILE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=MIN_NEIGHBORS_FACE,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    for (x, y, w, h) in (f2 if f2 is not None else []):
        faces.append((int(x), int(y), int(w), int(h), "profile"))

    # 横顔（反転して逆向きも拾う）
    gray_flip = cv2.flip(gray, 1)
    f3 = FACE_PROFILE.detectMultiScale(
        gray_flip,
        scaleFactor=1.1,
        minNeighbors=MIN_NEIGHBORS_FACE,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )
    if f3 is not None:
        w_img = gray.shape[1]
        for (x, y, w, h) in f3:
            # 反転座標を元に戻す
            x2 = w_img - (x + w)
            faces.append((int(x2), int(y), int(w), int(h), "profile_flip"))

    return faces


def score_candidate(x, y, w, h, img_w, img_h):
    """
    候補の優先順位付け
    - 基本は「面積が大きいほど優先」
    - ただし「下側にあるbbox」は強めに減点（胸/腹の誤検出を落とす）
    """
    area = w * h

    # bbox中心のY（0=上, img_h=下）
    cy = y + h / 2.0
    cy_norm = cy / img_h  # 0.0(上)〜1.0(下)

    # 下側ペナルティ：
    # 画像の55%より下に行くほど減点（顔が中央の写真を守るため "55%" から）
    lower_penalty = max(0.0, cy_norm - 0.55)

    # 減点の強さ（0.50〜0.90で調整）
    LOWER_PENALTY_WEIGHT = 0.75

    return area - area * LOWER_PENALTY_WEIGHT * lower_penalty


def validate_by_eyes(gray: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[bool, int]:
    """
    顔bboxの中に「目」があるかチェックする。
    - 指や胸や背景などの誤検出は目が出ないので落ちる
    - 横顔/伏し目/強いボケでは目検出が失敗して落ちやすい

    戻り値:
      (採用してよいか, 検出された目の数)
    """
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return False, 0

    eyes = EYE_CASCADE.detectMultiScale(
        roi,
        scaleFactor=1.1,
        minNeighbors=MIN_NEIGHBORS_EYE,
        minSize=(EYE_MIN_SIZE, EYE_MIN_SIZE),
    )
    if eyes is None or len(eyes) == 0:
        return False, 0

    # 目は顔bboxの上側にあるはず、という制約でさらに誤検出を減らす
    valid = 0
    y_limit = int(h * EYE_REGION_Y_MAX)
    for (ex, ey, ew, eh) in eyes:
        cy = ey + eh / 2.0
        if cy <= y_limit:
            valid += 1

    return (valid >= MIN_EYES), valid


def pick_best_face(gray: np.ndarray, img_w: int, img_h: int):
    """
    顔候補を収集し、スコア順に並べ、
    目検証（REQUIRE_EYES）を通った最初のものを採用する。

    戻り値:
      (x, y, w, h, kind, eye_count) or None

    顔候補を集めてベストを返す（2パス方式）

    1st: 目検証あり（誤検出を抑える）
    2nd: 目検証なし（顔があるのに no_face を減らす救済）
    """
    candidates = detect_faces(gray)
    if not candidates:
        return None

    candidates.sort(
        key=lambda f: score_candidate(f[0], f[1], f[2], f[3], img_w, img_h),
        reverse=True
    )

    # -------------------------
    # 1st pass: strict (eyes)
    # -------------------------
    if REQUIRE_EYES:
        for (x, y, w, h, kind) in candidates:
            ok, eye_count = validate_by_eyes(gray, x, y, w, h)
            if not ok:
                continue
            return (x, y, w, h, kind, eye_count)

    # -------------------------
    # 2nd pass: fallback (no eyes)
    # -------------------------
    # ここで拾う候補は「誤検出の可能性」もあるが、
    # score_candidate に入れた下側ペナルティが胸誤検出を負けやすくしてくれる
    for (x, y, w, h, kind) in candidates:
        return (x, y, w, h, kind, 0)

    return None


# ============================================================
# クロップ生成
# ============================================================

def expanded_square_crop_box(x, y, bw, bh, img_w, img_h, scale):
    """
    顔bbox中心に正方形のクロップ領域を作る。
    - 顔bboxの最大辺を基準にscale倍して余白を作る
    - 画像外にはみ出さないようclampする
    """
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    size = max(bw, bh) * scale

    left = int(cx - size / 2.0)
    top = int(cy - size / 2.0)
    right = int(cx + size / 2.0)
    bottom = int(cy + size / 2.0)

    left = clamp(left, 0, img_w - 1)
    top = clamp(top, 0, img_h - 1)
    right = clamp(right, 1, img_w)
    bottom = clamp(bottom, 1, img_h)

    if right - left < 2 or bottom - top < 2:
        return None
    return (left, top, right, bottom)


def save_square(pil_img: Image.Image, out_path: Path, size: int, allow_upscale: bool) -> str:
    """
    正方形にリサイズして保存する。
    - allow_upscale=False の場合、1024未満ならそのまま保存（ボケを避ける）
    """
    w, h = pil_img.size
    if not allow_upscale and (w < size or h < size):
        pil_img.save(out_path)
        return "no_upscale_saved_raw"
    resized = pil_img.resize((size, size), resample=Image.LANCZOS)
    resized.save(out_path)
    return "resized"


# ============================================================
# Main
# ============================================================

def main():
    """
    全体フロー:
      1) out/ を削除
      2) out/ 配下を作成
      3) input/ から画像を列挙
      4) 顔検出 + 目検証 + 小さすぎる顔をreject
      5) A/B/Cへコピー
      6) B/Cのみ（設定次第） cropped_1024 を生成
      7) report.csv を出力
    """
    reset_output_dir()
    ensure_dirs()

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [p for p in INPUT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print(f"No images found in: {INPUT_DIR.resolve()}")
        print("Put your images under: input\\")
        return

    rows = []
    total = 0
    no_face = 0
    crop_failed = 0
    rejected_small = 0

    for p in files:
        total += 1
        rel = p.relative_to(INPUT_DIR)
        stem = p.stem
        ext = p.suffix.lower()

        bgr = imread_bgr(p)
        if bgr is None:
            rows.append([str(rel), "read_failed", "", "", "", "", "", "", "", "A_fullbody", ""])
            continue

        img_h, img_w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        picked = pick_best_face(gray, img_w, img_h)
        if picked is None:
            # 顔が取れない -> A_fullbody に回す
            no_face += 1
            dst = OUT_DIR / "A_fullbody" / rel.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            rows.append([str(rel), "no_face", img_w, img_h, "", "", "", "", "", "A_fullbody", ""])
            continue

        x, y, bw, bh, kind, eye_count = picked

        # 顔占有率（face_area/img_area）
        face_ratio = (bw * bh) / (img_w * img_h) if (img_w * img_h) > 0 else 0.0

        # --- ここが追加した「顔占有率で弾く」処理 ---
        # 小物や背景の誤検出は face_ratio が小さいことが多い
        # ただし全身で顔が小さい写真も落ちる（仕様）
        if face_ratio < MIN_FACE_AREA_RATIO:
            rejected_small += 1
            no_face += 1
            dst = OUT_DIR / "A_fullbody" / rel.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            rows.append([
                str(rel),
                f"{face_ratio:.4f}",
                img_w, img_h,
                x, y, bw, bh,
                kind,
                "A_fullbody",
                f"reject_small_face;eyes={eye_count}"
            ])
            continue

        # 仕分け（face_ratioベース）
        if face_ratio < A_MAX:
            cls = "A_fullbody"
        elif face_ratio < B_MAX:
            cls = "B_upper"
        else:
            cls = "C_closeup"

        # 元画像をクラスフォルダへコピー
        dst = OUT_DIR / cls / rel.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

        # cropped_1024 の作成
        crop_note = ""
        if CROP_ONLY_FOR_BC and cls == "A_fullbody":
            # A_fullbody は学習に混ぜない（髪・背景学習を避ける）
            crop_note = "skip_crop_A_fullbody"
        else:
            crop_box = expanded_square_crop_box(x, y, bw, bh, img_w, img_h, CROP_SCALE)
            if crop_box is None:
                crop_failed += 1
                crop_note = "crop_failed"
            else:
                pil = pil_open_rgb(p)
                cropped = pil.crop(crop_box)
                out_crop = OUT_DIR / "cropped_1024" / f"{stem}_crop{ext}"
                out_crop.parent.mkdir(parents=True, exist_ok=True)
                crop_note = save_square(cropped, out_crop, OUTPUT_SIZE, ALLOW_UPSCALE)

        # report.csv にログ出し
        rows.append([
            str(rel),
            f"{face_ratio:.4f}",
            img_w, img_h,
            x, y, bw, bh,
            kind,
            cls,
            f"{crop_note};eyes={eye_count}"
        ])

    # report.csv 出力
    report_path = OUT_DIR / "report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "face_ratio", "img_w", "img_h", "x", "y", "bw", "bh", "detector", "class", "note"])
        writer.writerows(rows)

    print("Done.")
    print(f"- Images processed : {total}")
    print(f"- No face detected : {no_face}")
    print(f"- Rejected small   : {rejected_small}")
    print(f"- Crop failed      : {crop_failed}")
    print(f"- Output folder    : {OUT_DIR.resolve()}")
    print(f"- Report CSV       : {report_path.resolve()}")
    print(f"- Cropped dataset  : {(OUT_DIR / 'cropped_1024').resolve()}")
    print("")
    print("Tip:")
    print("- If too many are rejected, try MIN_FACE_AREA_RATIO=0.05.")
    print("- If too much hair is included, set CROP_SCALE=1.5.")
    print("- If many faces are missed, try MIN_FACE_SIZE=30 (more false positives).")


if __name__ == "__main__":
    main()

