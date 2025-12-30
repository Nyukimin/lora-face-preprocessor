# MCP設定の有効/無効化の例

このファイルでは、プロジェクトごとにMCPサーバーを有効/無効化する設定例を示します。

## 設定パターン

### パターン1: 両方有効

`.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "blender-mcp": {
      "command": "python",
      "args": [
        "-m",
        "blender_mcp.server"
      ],
      "cwd": "E:\\GenerativeAI\\MCP\\blender-mcp"
    },
    "serena": {
      "command": "python",
      "args": [
        "-m",
        "serena.cli",
        "start_mcp_server"
      ],
      "cwd": "E:\\GenerativeAI\\MCP\\serena"
    }
  }
}
```

### パターン2: Blenderのみ有効

`.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "blender-mcp": {
      "command": "python",
      "args": [
        "-m",
        "blender_mcp.server"
      ],
      "cwd": "E:\\GenerativeAI\\MCP\\blender-mcp"
    }
  }
}
```

### パターン3: Serenaのみ有効

`.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "serena": {
      "command": "python",
      "args": [
        "-m",
        "serena.cli",
        "start_mcp_server"
      ],
      "cwd": "E:\\GenerativeAI\\MCP\\serena"
    }
  }
}
```

### パターン4: 両方無効（プロジェクト設定のみ）

`.cursor/mcp.json`:
```json
{
  "mcpServers": {}
}
```

この場合、グローバル設定（`C:\Users\nyuki\.cursor\mcp.json`）が適用されます。

## 設定変更後の手順

1. `.cursor/mcp.json`を編集
2. Cursor IDEを再起動
3. MCPサーバーが正しく起動しているか確認

## トラブルシューティング

### グローバル設定を完全に無効化したい場合

プロジェクト設定で空の`mcpServers`を定義しても、グローバル設定が適用される可能性があります。
その場合は、グローバル設定（`C:\Users\nyuki\.cursor\mcp.json`）を直接編集してください。

### 設定が反映されない場合

1. JSON構文エラーがないか確認（カンマ、括弧の位置など）
2. Cursor IDEを完全に再起動
3. CursorのMCP設定UIからも確認

