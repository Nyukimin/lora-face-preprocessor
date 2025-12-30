#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.csvの詳細分析と問題画像の特定
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

# Windowsでのエンコーディング問題を回避
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_detailed(csv_path):
    """report.csvを詳細分析"""
    report_path = Path(csv_path)
    if not report_path.exists():
        print(f"エラー: {csv_path} が見つかりません")
        return
    
    rows = []
    with open(report_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    total = len(rows)
    
    # カテゴリ別に分類
    no_face = []
    rejected_small = []
    saved = []
    face_ratios = []
    
    detector_stats = defaultdict(int)
    class_stats = defaultdict(int)
    
    for row in rows:
        if row['face_ratio'] == 'no_face':
            no_face.append(row)
        else:
            try:
                face_ratio = float(row['face_ratio'])
                face_ratios.append(face_ratio)
                
                note = row.get('note', '')
                if 'reject_small_face' in note:
                    rejected_small.append(row)
                elif 'no_upscale_saved_raw' in note:
                    saved.append(row)
                
                detector_stats[row.get('detector', 'unknown')] += 1
                class_stats[row.get('class', 'unknown')] += 1
            except ValueError:
                pass
    
    # 詳細統計を表示
    print("=" * 70)
    print("詳細分析レポート")
    print("=" * 70)
    print(f"\n総画像数: {total}")
    print(f"顔検出: {len(face_ratios)} ({len(face_ratios)/total*100:.1f}%)")
    print(f"検出なし: {len(no_face)} ({len(no_face)/total*100:.1f}%)")
    print(f"保存済み: {len(saved)} ({len(saved)/total*100:.1f}%)")
    print(f"リジェクト: {len(rejected_small)} ({len(rejected_small)/total*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("検出器別統計")
    print("=" * 70)
    for detector, count in sorted(detector_stats.items(), key=lambda x: -x[1]):
        print(f"  {detector}: {count}件")
    
    print("\n" + "=" * 70)
    print("分類別統計")
    print("=" * 70)
    for cls, count in sorted(class_stats.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}件")
    
    # 顔検出されなかった画像のリスト
    if no_face:
        print("\n" + "=" * 70)
        print(f"顔検出されなかった画像 ({len(no_face)}件)")
        print("=" * 70)
        for i, row in enumerate(no_face[:20], 1):  # 最大20件表示
            img_size = f"{row.get('img_w', '?')}x{row.get('img_h', '?')}"
            print(f"  {i:2d}. {row['file']} ({img_size})")
        if len(no_face) > 20:
            print(f"  ... 他 {len(no_face) - 20}件")
    
    # 小さな顔でリジェクトされた画像
    if rejected_small:
        print("\n" + "=" * 70)
        print(f"小さな顔でリジェクトされた画像 ({len(rejected_small)}件)")
        print("=" * 70)
        # 顔比率でソート
        rejected_sorted = sorted(
            rejected_small,
            key=lambda r: float(r['face_ratio']) if r['face_ratio'] != 'no_face' else 0
        )
        for i, row in enumerate(rejected_sorted[:20], 1):  # 最大20件表示
            face_ratio = row['face_ratio']
            eyes = row.get('note', '').split('eyes=')[1].split(';')[0] if 'eyes=' in row.get('note', '') else '?'
            print(f"  {i:2d}. {row['file']} (face_ratio={face_ratio}, eyes={eyes})")
        if len(rejected_small) > 20:
            print(f"  ... 他 {len(rejected_small) - 20}件")
    
    # 正常に保存された画像の統計
    if saved:
        print("\n" + "=" * 70)
        print(f"正常に保存された画像 ({len(saved)}件)")
        print("=" * 70)
        saved_ratios = [float(r['face_ratio']) for r in saved if r['face_ratio'] != 'no_face']
        if saved_ratios:
            print(f"  平均顔比率: {sum(saved_ratios)/len(saved_ratios):.4f}")
            print(f"  最小: {min(saved_ratios):.4f}")
            print(f"  最大: {max(saved_ratios):.4f}")
        
        # 分類別の保存数
        saved_by_class = defaultdict(int)
        for r in saved:
            saved_by_class[r.get('class', 'unknown')] += 1
        print("\n  分類別保存数:")
        for cls, count in sorted(saved_by_class.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}件")
    
    # 改善提案
    print("\n" + "=" * 70)
    print("改善提案")
    print("=" * 70)
    
    detection_rate = len(face_ratios) / total if total > 0 else 0
    rejection_rate = len(rejected_small) / len(face_ratios) if face_ratios else 0
    save_rate = len(saved) / total if total > 0 else 0
    
    if detection_rate < 0.5:
        print(f"\n[問題] 顔検出率が低い ({detection_rate*100:.1f}%)")
        print("  推奨設定:")
        print("    MIN_FACE_SIZE = 30  (現在の設定を確認)")
        print("    MIN_NEIGHBORS_FACE = 3  (現在の設定を確認)")
        print("    REQUIRE_EYES = False  (横顔も拾う)")
    
    if rejection_rate > 0.4:
        print(f"\n[問題] リジェクト率が高い ({rejection_rate*100:.1f}%)")
        print("  推奨設定:")
        print("    MIN_FACE_AREA_RATIO = 0.05  (現在の設定を確認)")
    
    if save_rate < 0.3:
        print(f"\n[問題] 保存率が低い ({save_rate*100:.1f}%)")
        print("  推奨設定:")
        print("    MIN_FACE_AREA_RATIO = 0.05")
        print("    MIN_FACE_SIZE = 30")
        print("    REQUIRE_EYES = False")
        print("    MIN_EYES = 1")
    
    if detection_rate >= 0.5 and save_rate >= 0.3:
        print("\n[良好] 現在の設定で良好な結果が得られています")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_detailed("out/report.csv")

