"""Tests for the minimal MCP server."""

import json
from io import StringIO
from unittest.mock import patch

from imessage_rag import mcp_server


class TestHandleRequest:
    def test_initialize(self):
        result = mcp_server._handle_request(
            "initialize",
            {"protocolVersion": "2025-06-18"},
        )
        assert result["protocolVersion"] == "2025-06-18"
        assert result["capabilities"] == {"tools": {}}
        assert result["serverInfo"]["name"] == "personal-rag"

    def test_tools_list(self):
        result = mcp_server._handle_request("tools/list", {})
        names = {tool["name"] for tool in result["tools"]}
        assert names == {"search_messages", "get_chunk", "get_stats"}

    @patch("imessage_rag.mcp_server._search_messages")
    def test_tools_call_dispatches(self, mock_search):
        mock_search.return_value = {"content": [], "structuredContent": {"results": []}}
        result = mcp_server._handle_request(
            "tools/call",
            {"name": "search_messages", "arguments": {"query": "dinner"}},
        )
        assert result["structuredContent"] == {"results": []}
        mock_search.assert_called_once_with({"query": "dinner"})

    def test_unknown_method_raises(self):
        try:
            mcp_server._handle_request("nope", {})
            assert False, "expected JsonRpcError"
        except mcp_server.JsonRpcError as exc:
            assert exc.code == -32601


class TestServerLoop:
    def test_serve_handles_ping(self):
        inp = StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n")
        out = StringIO()
        rc = mcp_server.serve(inp, out)
        assert rc == 0
        payload = json.loads(out.getvalue().strip())
        assert payload["id"] == 1
        assert payload["result"] == {}

    def test_serve_handles_parse_error(self):
        inp = StringIO("{bad json}\n")
        out = StringIO()
        mcp_server.serve(inp, out)
        payload = json.loads(out.getvalue().strip())
        assert payload["error"]["code"] == -32700
