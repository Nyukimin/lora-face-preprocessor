#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.csvを分析して統計情報を表示
"""

import csv
import sys
from pathlib import Path

# Windowsでのエンコーディング問題を回避
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_report(csv_path):
    """report.csvを分析"""
    report_path = Path(csv_path)
    if not report_path.exists():
        print(f"エラー: {csv_path} が見つかりません")
        return
    
    total = 0
    no_face = 0
    rejected_small = 0
    saved_raw = 0
    face_ratios = []
    eye_counts = []
    
    with open(report_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            
            # 顔検出なし
            if row['face_ratio'] == 'no_face':
                no_face += 1
                continue
            
            # 顔比率を数値に変換
            try:
                face_ratio = float(row['face_ratio'])
                face_ratios.append(face_ratio)
            except ValueError:
                continue
            
            # ノートから情報を抽出
            note = row.get('note', '')
            if 'reject_small_face' in note:
                rejected_small += 1
            elif 'no_upscale_saved_raw' in note:
                saved_raw += 1
            
            # 目の数を抽出
            if 'eyes=' in note:
                try:
                    eyes = int(note.split('eyes=')[1].split(';')[0])
                    eye_counts.append(eyes)
                except (IndexError, ValueError):
                    pass
    
    # 統計を表示
    print("=" * 60)
    print("[統計] 処理結果")
    print("=" * 60)
    print(f"総画像数: {total}")
    print(f"")
    print("【検出結果】")
    detected = total - no_face
    print(f"  顔検出: {detected} ({detected/total*100:.1f}%)")
    print(f"  検出なし: {no_face} ({no_face/total*100:.1f}%)")
    print(f"")
    print("【処理結果】")
    print(f"  正常処理（保存）: {saved_raw} ({saved_raw/total*100:.1f}%)")
    print(f"  小さな顔でリジェクト: {rejected_small} ({rejected_small/total*100:.1f}%)")
    print(f"")
    
    if face_ratios:
        print("【顔サイズ統計】")
        print(f"  平均顔比率: {sum(face_ratios)/len(face_ratios):.4f}")
        print(f"  最小: {min(face_ratios):.4f}")
        print(f"  最大: {max(face_ratios):.4f}")
        print(f"")
        
        # 分類別の統計
        a_count = sum(1 for r in face_ratios if r < 0.05)
        b_count = sum(1 for r in face_ratios if 0.05 <= r < 0.15)
        c_count = sum(1 for r in face_ratios if 0.15 <= r)
        
        print("【分類別（検出されたもののみ）】")
        print(f"  A_fullbody (<0.05): {a_count}")
        print(f"  B_upper (0.05-0.15): {b_count}")
        print(f"  C_closeup (>=0.15): {c_count}")
        print(f"")
    
    if eye_counts:
        print("【目検出統計】")
        eye_dist = {}
        for eyes in eye_counts:
            eye_dist[eyes] = eye_dist.get(eyes, 0) + 1
        for eyes in sorted(eye_dist.keys()):
            print(f"  {eyes}個: {eye_dist[eyes]}件")
        print(f"")
    
    print("=" * 60)
    print("[提案] 改善案")
    print("=" * 60)
    
    if no_face / total > 0.5:
        print("[警告] 顔検出率が低いです（50%以下）")
        print("   → MIN_FACE_SIZE を小さくする（例: 30）")
        print("   → MIN_NEIGHBORS_FACE を小さくする（例: 3）")
        print("")
    
    if rejected_small / detected > 0.5 if detected > 0 else False:
        print("[警告] 小さな顔でリジェクトが多いです")
        print("   → MIN_FACE_AREA_RATIO を小さくする（例: 0.05）")
        print("")
    
    if saved_raw < 10:
        print("[警告] 保存された画像が少ないです（10枚未満）")
        print("   → MIN_FACE_AREA_RATIO を小さくする")
        print("   → MIN_FACE_SIZE を小さくする")
        print("   → REQUIRE_EYES を False にする（横顔も拾う）")
        print("")

if __name__ == "__main__":
    analyze_report("out/report.csv")

