# Cursor MCP設定

このディレクトリには、Cursor IDEのMCP（Model Context Protocol）設定が含まれています。

## 設定ファイル

- `mcp.json`: MCPサーバーの設定ファイル（このプロジェクト専用）
- `mcp.json.template`: MCP設定のテンプレートファイル

## このプロジェクトで使用可能なMCPサーバー

1. **blender-mcp**: Blender統合MCPサーバー
   - パス: `E:\GenerativeAI\MCP\blender-mcp`
   - エントリーポイント: `blender_mcp.server`

2. **serena**: Serena MCPサーバー
   - パス: `E:\GenerativeAI\MCP\serena`
   - エントリーポイント: `serena.cli start_mcp_server`

## 有効/無効の設定方法

### プロジェクトごとにMCPサーバーを有効/無効化する

`mcp.json`ファイルを編集して、必要なMCPサーバーを有効化または無効化できます。

**有効化する場合:**
- `mcpServers`オブジェクト内に対応するエントリを追加または残します

**無効化する場合:**
- `mcpServers`オブジェクト内から該当するエントリ全体を削除します

### 設定例

**両方有効な場合:**
```json
{
  "mcpServers": {
    "blender-mcp": { ... },
    "serena": { ... }
  }
}
```

**Blenderのみ有効な場合:**
```json
{
  "mcpServers": {
    "blender-mcp": { ... }
  }
}
```

**Serenaのみ有効な場合:**
```json
{
  "mcpServers": {
    "serena": { ... }
  }
}
```

**両方無効な場合:**
```json
{
  "mcpServers": {}
}
```

## 使用方法

1. `mcp.json`を編集して、必要なMCPサーバーを有効化してください
2. 無効化したいサーバーは、該当するエントリ全体を削除してください
3. 各サーバーの設定（コマンド、引数、環境変数）を必要に応じて調整してください
4. APIキーなどの機密情報は環境変数ファイル（`.env`）を使用することを推奨します

## グローバル設定との関係

- グローバル設定（`C:\Users\nyuki\.cursor\mcp.json`）は全プロジェクトに適用されます
- プロジェクト固有の設定（`.cursor/mcp.json`）は、このプロジェクトでのみ有効です
- プロジェクト設定で定義されたサーバーは、グローバル設定を上書きします
- プロジェクト設定に含まれていないサーバーは、グローバル設定が適用されます

## 環境変数の管理

機密情報（APIキーなど）を安全に管理するには、`mcp.json`で`envFile`パラメータを使用できます：

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "mcp-server"],
      "envFile": ".env"
    }
  }
}
```

`.env`ファイルは`.gitignore`に追加することを忘れずに！

## 参考リンク

- [Cursor MCP Documentation](https://docs.cursor.com/context/mcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
