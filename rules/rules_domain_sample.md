# rules_domain.md - loganalyzer ドメイン固有ルール（サンプル）

**作成日**: 2025-12-11  
**プロジェクト名**: loganalyzer  
**目的**: loganalyzer プロジェクト固有のドメイン知識・ルール  
※このファイルは「例＋記入ガイド」です。自分のプロジェクトに合わせて書き換えてください。

---

## 1. Chrome DevTools MCP 使用ガイド

> ガイド: ここには「このプロジェクトで、ブラウザをどう触るか」の運転マニュアルを書きます。  
> 危険な操作の禁止、よく使うツールの組み合わせ、データ量が多いときの方針などをまとめます。

### 1.1 大量データのデバッグ戦略

> ガイド: 1,000件以上の要素や、巨大なテーブルを扱うときの「基本姿勢」を書きます。  
> 「何をしてはいけないか」「代わりに何を使うか」をはっきりさせるのが目的です。

#### DOM 出力制御

> ガイド: DOM をそのまま全部出すと落ちるので、「出し過ぎ防止ルール」と「例外条件」をここで定義します。

##### 基本制限

- **デフォルト制限**: 5,000 文字
- **省略表記**: `[DOM content omitted - {size} characters. Use evaluate_script or uid-specific snapshot for details]`
- **適用対象ツール**:
  - `mcp__Chrome-DevTools-MCP__take_snapshot`（全体スナップショット）
  - `mcp__Chrome-DevTools-MCP__list_console_messages`（大量ログ時）
  - `mcp__Chrome-DevTools-MCP__list_network_requests`（大量リクエスト時）

##### 例外条件（制限を適用しない場合）

> ガイド: 「本当に全部欲しいとき」や「ピンポイント取得」のような、制限を外してよい条件を書きます。

1. **ユーザー明示的要求**
   - "完全なDOMを見せて"
   - "全HTMLを出力"
   - "制限なしでスナップショット"
   - "すべての内容を表示"

2. **uid 指定の要素取得**
   - 特定要素のみを取得する場合は制限なし

3. **evaluate_script の実行結果**
   - カスタムスクリプトの結果は制限なし

### 1.2 大量データ（1,000件以上）の推奨アプローチ

> ガイド: 「とりあえず全部取る」は禁止、という前提で、段階的にどう調べるかの手順を書きます。  
> ここを読めば、誰でも同じ手順でデバッグできる状態を目指します。

大量のデータ（テーブル、リスト、グリッドなど）をデバッグする際は、以下の段階的アプローチを推奨します。

#### Phase 1: 概要把握

**目的**: 全体構造とデータ量の把握

1. **視覚的確認**
   ```text
   take_screenshot: レイアウト、スタイル、表示状態の確認
   ```

2. **データ統計取得**
   ```javascript
   evaluate_script:
   () => {
     const items = document.querySelectorAll('.data-item'); // セレクタは適宜調整
     const errors = document.querySelectorAll('.error, [data-error="true"]');
     return {
       total_items: items.length,
       error_count: errors.length,
       page_structure: {
         containers: document.querySelectorAll('[data-container]').length,
         visible_items: Array.from(items).filter(el => 
           el.offsetParent !== null
         ).length
       }
     };
   }
   ```

3. **コンソールエラー確認**
   ```text
   list_console_messages: JavaScriptエラーの有無確認
   ```

#### Phase 2: 問題の特定

**目的**: エラーや異常のある要素を特定

1. **エラー要素の uid 抽出**
   ```javascript
   evaluate_script:
   () => {
     const errorElements = document.querySelectorAll('.error');
     return Array.from(errorElements).slice(0, 10).map(el => ({
       uid: el.getAttribute('data-uid') || el.id,
       text: el.textContent.slice(0, 100),
       classList: Array.from(el.classList)
     }));
   }
   ```

2. **個別要素の詳細確認**
   ```text
   take_snapshot（uid 指定）: 特定のエラー要素を詳細に確認
   ```

#### Phase 3: パターン分析

**目的**: 問題の共通パターンを発見

1. **サンプリング**
   ```javascript
   evaluate_script:
   () => {
     const items = document.querySelectorAll('.data-item');
     const total = items.length;
     // 最初、中間、最後からサンプル抽出
     const samples = [
       ...Array.from(items).slice(0, 5),
       ...Array.from(items).slice(Math.floor(total/2), Math.floor(total/2) + 5),
       ...Array.from(items).slice(-5)
     ];
     return samples.map(el => ({
       position: Array.from(items).indexOf(el),
       content: el.textContent.slice(0, 100),
       hasError: el.classList.contains('error'),
       attributes: Object.fromEntries(
         Array.from(el.attributes).map(a => [a.name, a.value])
       )
     }));
   }
   ```

2. **複数ポイントでの検証**
   - サンプル結果から共通パターンを抽出
   - 必要に応じて特定の uid で詳細確認

#### Phase 4: 修正と検証

1. コード修正
2. Phase 1〜3 を再実行して問題解決を確認

### 1.3 実践例: 10,000件のテーブルデバッグ

> ガイド: 実際にありそうなケースを 1 つだけ、ストーリー形式で残しておくと便利です。  
> ログ的な「デバッグ日誌」は docs/調査/ 側に回し、ここには再利用したいパターンだけを書きます。

```markdown
問題: 10,000 行のテーブルで一部の行が正しく表示されない

Step 1: スクリーンショット + 統計スクリプト
→ 結果: 9,850 行が正常、150 行がエラー

Step 2: エラー行の uid 抽出スクリプト
→ 結果: 最初のエラー行の uid を取得

Step 3: uid 指定でスナップショット
→ 結果: データ属性が欠落していることを発見

Step 4: パターン検証のためサンプリング
→ 結果: すべてのエラー行で同じ属性が欠落

Step 5: コード修正・再検証
```

### 1.4 ツール使い分けガイド

> ガイド: 「何をしたいときに、どのツールを使うか」を表で固定します。  
> ここはプロジェクトごとにかなり内容が変わるので、実態に合わせて書き換えてください。

| 目的 | 推奨ツール | DOM制限 | 備考 |
|------|-----------|---------|------|
| 視覚的確認 | `take_screenshot` | なし | レイアウト、スタイル確認 |
| 全体構造把握 | `take_snapshot`（全体） | 5,000文字 | 小規模ページのみ |
| 統計・カウント | `evaluate_script` | なし | カスタムデータ抽出 |
| 特定要素詳細 | `take_snapshot`（uid） | なし | 問題箇所の詳細確認 |
| エラーログ | `list_console_messages` | 5,000文字 | JavaScript 実行エラー |
| ネットワーク | `list_network_requests` | 5,000文字 | API 呼び出し確認 |

### 1.5 DOM 制限を超えた場合の対応

> ガイド: 「制限に引っかかったとき、AI がどう振る舞うか」をここで決めておきます。

5,000 文字を超える DOM 出力が返された場合:

1. **自動的に省略**
   - `[DOM content omitted - 25,000 characters. Use evaluate_script or uid-specific snapshot for details]`
   - データ件数、主要構造のサマリーのみ表示

2. **次のアクション提案**
   - `evaluate_script` でカスタムデータ抽出
   - 特定要素の `uid` を特定して詳細確認
   - スクリーンショットで視覚的確認

3. **ユーザーへの確認**
   - "完全な DOM が必要ですか？"
   - "特定の要素を詳しく見ますか？"

### 1.6 システム安定性の維持

> ガイド: 「これを守らないとブラウザごと落ちる」系の原則をまとめます。

#### 大量データ処理時の原則

1. **段階的アプローチ**: 一度にすべてを取得しない
2. **必要最小限の情報**: 目的に応じた適切なツール選択
3. **サンプリング**: 代表的なデータで検証
4. **スクリプト活用**: DOM そのものではなく、必要なデータだけを抽出

#### パフォーマンス考慮

- 10,000 件以上のデータ: 必ず `evaluate_script` でサンプリング
- 全件スナップショットは避ける
- ページネーションや仮想スクロールなど、画面側の仕組みも考慮して調査する

---

## 2. ログファイル配置ルール（詳細）

> ガイド: 「ログがどこにあるか分からない」問題を潰すためのセクションです。  
> 出力ディレクトリ、命名規則、ツールの入口をここで固定します。

### 2.1 ログ出力場所の詳細

すべてのログファイルは **必ず** プロジェクトルートの `logs/` ディレクトリに保存すること。

#### 禁止事項

- 他のディレクトリ（`procedures/logs/`、`./logs` など）への出力は禁止
- スクリプト内で出力ディレクトリを指定する場合は、必ずプロジェクトルートの `logs/` を指すように設定

#### デフォルト出力ディレクトリの設定

- `${SCRIPT_DIR}/../../logs` または `./logs`（プロジェクトルートからの相対パス）

### 2.2 ディレクトリ命名規則の詳細

> ガイド: 「同じ日に複数ホストから集めても迷子にならない」命名規則をここに書きます。

#### 形式

```text
${HOSTNAME}_${TIMESTAMP}
```

#### 例

```text
192.168.1.17_20251209_103532
```

#### パラメータ

- **`HOSTNAME`**: リモートホストの IP アドレスまたはホスト名  
  - `:` は `_` に変換（例: `192.168.1.17:22` → `192.168.1.17_22`）

- **`TIMESTAMP`**: 取得日時  
  - 形式: `YYYYMMDD_HHMMSS`  
  - タイムゾーン: JST（日本標準時）

### 2.3 ログ取得・処理の基本ツールの使用方法

> ガイド: 「どのスクリプトから触るべきか」「どれが正しい入口か」をここに明記します。

#### 基本ツール

- **`procedures/tools/get_data.sh`**: ログ取得・マージ・タイムスタンプ調整を一括実行
- **`procedures/tools/split_log_file.py`**: ログファイルの分割・時刻範囲フィルタリング

#### 使用方法

これらのツールを組み合わせて使用することを推奨します。  
詳細な使用方法は各スクリプトのヘルプまたは `logs/README.md` を参照してください。

---

## 3. ログ分析プロジェクト固有の原則

> ガイド: 「ログ分析ならではのこだわり」を書く場所です。  
> タイムゾーン、ソート順、欠損値の扱いなど、分析品質に直結するルールをここにまとめます。

**注意**: データ処理の基本原則（データの完全性、時間精度、冪等性、エラーハンドリング、ドキュメント同期）については、  
`../common/GLOBAL_AGENT.md` の「2.2 作業の進め方 > データ処理の基本原則」を参照してください。

### 3.1 タイムゾーンの扱い

- ログファイルのタイムスタンプは原則 JST（日本標準時）で扱う
- 別タイムゾーンのログを扱う場合は、変換前後のタイムゾーンを必ず明記すること
- タイムスタンプの変換や調整を行う際は、タイムゾーンを明示的に指定すること

### 3.2 ログファイルのマージ・分割

- 複数のログファイルをマージする際は、タイムスタンプでソートすること
- ログファイルを分割する際は、時刻範囲を明確に指定すること
- マージ・分割の条件は、可能であれば `docs/調査/` にサンプルとして残しておく

---

**最終更新**: 2025-12-11  
**バージョン**: 1.0  
※ルールを変更した場合は、このファイルと `PROJECT_AGENT.md` の両方を更新してください。
