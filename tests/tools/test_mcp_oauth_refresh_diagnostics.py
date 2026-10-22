"""Refresh diagnostics expose only allowlisted OAuth codes, never provider text."""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest

from tools.mcp_oauth_provider import HermesProviderMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,expected", [
    ({"error": "invalid_grant", "error_description": "secret-description", "refresh_token": "secret-token"}, "invalid_grant"),
    ({"error": "invalid_client"}, "invalid_client"),
    ({"error": "temporarily_unavailable"}, "temporarily_unavailable"),
    ({"error": "secret-token"}, "unknown"),
    ({"error": ["invalid_grant"]}, "unknown"),
    ({"error": {"invalid_grant": "secret-token"}}, "unknown"),
    ({"error_description": "invalid_grant secret-description"}, "unknown"),
    (["invalid_grant"], "unknown"),
])
async def test_refresh_rejection_logs_only_allowlisted_error(payload, expected, caplog):
    provider = HermesProviderMixin.__new__(HermesProviderMixin)
    provider._hermes_logger = logging.getLogger("test.oauth.diagnostics")
    provider.context = SimpleNamespace(clear_tokens=lambda: None)
    provider._hermes_reload_tokens_after_refresh_failure = AsyncMock(return_value=False)
    provider._hermes_clear_persisted_tokens = AsyncMock()
    response = httpx2.Response(400, json=payload)
    with caplog.at_level(logging.WARNING):
        assert await provider._hermes_handle_refresh_response(response) is False
    assert f"Token refresh failed: 400 (oauth_error={expected})" in caplog.text
    assert "secret-token" not in caplog.text
    assert "secret-description" not in caplog.text
    assert "error_description" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not-json secret-token", b"<html>invalid_grant secret-token</html>"])
async def test_refresh_non_json_diagnostic_is_unknown(body):
    from tools.mcp_oauth_provider import _refresh_error_code
    assert await _refresh_error_code(httpx2.Response(400, content=body)) == "unknown"


@pytest.mark.asyncio
async def test_refresh_diagnostic_read_failure_does_not_mask_rejection():
    from tools.mcp_oauth_provider import _refresh_error_code
    response = SimpleNamespace(aread=AsyncMock(side_effect=RuntimeError("secret-token")))
    assert await _refresh_error_code(response) == "unknown"


@pytest.mark.asyncio
async def test_refresh_diagnostic_does_not_parse_oversized_body():
    from tools.mcp_oauth_provider import _refresh_error_code
    response = httpx2.Response(400, content=b'{"error":"invalid_grant","padding":"' + b'x' * 16384 + b'"}')
    assert await _refresh_error_code(response) == "unknown"


@pytest.mark.asyncio
async def test_refresh_diagnostic_propagates_cancellation():
    import asyncio
    from tools.mcp_oauth_provider import _refresh_error_code
    response = SimpleNamespace(aread=AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await _refresh_error_code(response)