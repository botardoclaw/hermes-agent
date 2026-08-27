"""Tests for the per-server ``oauth.user_agent`` on MCP OAuth token requests.

Some authorization servers and WAFs reject httpx's default User-Agent on the
token endpoint (#75576). The header is opt-in, per-server, and applied ONLY to
the two token-endpoint requests (authorization-code exchange and refresh) —
never to MCP traffic or discovery.

The tests drive the REAL provider classes' request builders end to end: the
``httpx.Request`` the SDK would send is what gets inspected, not a mocked
constructor call.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK required for OAuth support",
)

from tools.mcp_oauth import (  # noqa: E402 — after the SDK availability gate
    apply_oauth_provider_defaults,
    build_oauth_auth,
    token_request_user_agent,
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


@pytest.fixture(autouse=True)
def clean_port_state():
    import tools.mcp_oauth as mod

    mod._assigned_cimd_ports.clear()
    yield
    mod._assigned_cimd_ports.clear()
    for port in list(mod._reserved_sockets):
        sock = mod._reserved_sockets.pop(port, None)
        if sock is not None:
            sock.close()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_configured_user_agent_is_returned():
    assert token_request_user_agent({"user_agent": "My-MCP-Client/1.0"}) == "My-MCP-Client/1.0"


@pytest.mark.parametrize("cfg", [
    pytest.param({}, id="absent"),
    pytest.param({"user_agent": None}, id="null"),
    pytest.param({"user_agent": ""}, id="empty"),
    pytest.param({"user_agent": "   "}, id="whitespace-only"),
    pytest.param({"user_agent": 7}, id="non-string"),
])
def test_unset_user_agent_values_are_treated_as_absent(cfg):
    assert token_request_user_agent(cfg) is None


def test_user_agent_is_stripped():
    assert token_request_user_agent({"user_agent": "  UA/2 "}) == "UA/2"


def test_ibkr_provider_defaults_are_safe_and_overrideable():
    cfg = {}
    apply_oauth_provider_defaults(
        cfg,
        server_name="ibkr",
        server_url="https://api.ibkr.com/v1/api/mcp-public",
    )
    assert cfg["scope"] == "mcp.read"
    assert cfg["user_agent"] == "Hermes-Agent"

    explicit = {"scope": "mcp.write", "user_agent": "Custom-UA/9"}
    apply_oauth_provider_defaults(
        explicit,
        server_name="ibkr",
        server_url="https://api.ibkr.com/v1/api/mcp-public",
    )
    assert explicit["scope"] == "mcp.write"
    assert explicit["user_agent"] == "Custom-UA/9"


# ---------------------------------------------------------------------------
# The requests the SDK actually sends
# ---------------------------------------------------------------------------


def _ready_for_token_requests(provider):
    """Give the provider the minimum context both builders require."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    provider.context.oauth_metadata = SimpleNamespace(
        token_endpoint="https://idp.example.com/oauth/token"
    )
    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "client-1",
        "redirect_uris": ["http://127.0.0.1:33333/callback"],
    })
    provider.context.current_tokens = OAuthToken.model_validate({
        "access_token": "at",
        "token_type": "Bearer",
        "refresh_token": "rt",
    })


def _build_provider_via(builder, monkeypatch, tmp_path, cfg):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    return builder("srv", "https://mcp.example.com/mcp", cfg)


def _manager_builder(server_name, server_url, cfg):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()
    return MCPOAuthManager().get_or_build_provider(server_name, server_url, cfg)


@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
def test_token_requests_carry_the_configured_user_agent(
    builder, tmp_path, monkeypatch
):
    """Both token-endpoint requests, on both provider construction paths."""
    provider = _build_provider_via(
        builder, monkeypatch, tmp_path, {"user_agent": "My-MCP-Client/1.0"}
    )
    _ready_for_token_requests(provider)

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )
    refresh = asyncio.run(provider._refresh_token())

    assert exchange.headers["User-Agent"] == "My-MCP-Client/1.0"
    assert refresh.headers["User-Agent"] == "My-MCP-Client/1.0"


@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
def test_unconfigured_user_agent_leaves_the_default_header(
    builder, tmp_path, monkeypatch
):
    """No config → httpx's own default, exactly as before the feature."""
    import httpx

    provider = _build_provider_via(builder, monkeypatch, tmp_path, {})
    _ready_for_token_requests(provider)

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )
    refresh = asyncio.run(provider._refresh_token())

    default_ua = httpx.Request("POST", "https://x.example/").headers.get("User-Agent")
    assert exchange.headers.get("User-Agent") == default_ua
    assert refresh.headers.get("User-Agent") == default_ua


def test_user_agent_does_not_disturb_token_auth_preparation(tmp_path, monkeypatch):
    """The stamp runs after prepare_token_auth — a confidential client's
    Authorization header must survive alongside the custom User-Agent."""
    provider = _build_provider_via(
        build_oauth_auth, monkeypatch, tmp_path,
        {"user_agent": "UA/1", "client_id": "pre", "client_secret": "shh",
         "token_endpoint_auth_method": "client_secret_basic"},
    )
    _ready_for_token_requests(provider)
    from mcp.shared.auth import OAuthClientInformationFull

    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "pre",
        "client_secret": "shh",
        "token_endpoint_auth_method": "client_secret_basic",
        "redirect_uris": ["http://127.0.0.1:33333/callback"],
    })

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )

    assert exchange.headers["User-Agent"] == "UA/1"
    assert exchange.headers.get("Authorization", "").startswith("Basic ")


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
async def test_ibkr_oauth_discovery_and_registration_requests_carry_user_agent(
    builder, tmp_path, monkeypatch
):
    """IBKR needs the OAuth side-channel requests stamped with a stable UA.

    The initial MCP probe stays untouched; the yielded metadata + registration
    requests inside the SDK's 401 branch must carry Hermes' per-provider UA.
    """
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    assert httpx is not None

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    provider = builder(
        "ibkr",
        "https://api.ibkr.com/v1/api/mcp-public",
        {},
    )

    req = httpx.Request("POST", "https://api.ibkr.com/v1/api/mcp-public")
    flow = provider.async_auth_flow(req)

    outbound = await flow.__anext__()
    assert str(outbound.url) == "https://api.ibkr.com/v1/api/mcp-public"

    fake_401 = httpx.Response(
        401,
        request=outbound,
        headers={
            "www-authenticate": (
                'Bearer resource_metadata="'
                'https://api.ibkr.com/v1/api/mcp-public/.well-known/oauth-protected-resource"'
            )
        },
    )

    prm_req = await flow.asend(fake_401)
    assert str(prm_req.url).endswith("/.well-known/oauth-protected-resource")
    assert prm_req.headers.get("User-Agent") == "Hermes-Agent"

    prm_resp = httpx.Response(
        200,
        request=prm_req,
        headers={"content-type": "application/json"},
        content=json.dumps({
            "resource": "https://api.ibkr.com/v1/api/mcp-public",
            "authorization_servers": ["https://api.ibkr.com/oauth2"],
            "scopes_supported": ["mcp.read", "mcp.write"],
            "bearer_methods_supported": ["header"],
        }).encode(),
    )

    asm_req = await flow.asend(prm_resp)
    assert "/.well-known/" in str(asm_req.url)
    assert asm_req.headers.get("User-Agent") == "Hermes-Agent"

    asm_resp = httpx.Response(
        200,
        request=asm_req,
        headers={"content-type": "application/json"},
        content=json.dumps({
            "issuer": "https://api.ibkr.com",
            "authorization_endpoint": "https://api.ibkr.com/oauth2/authorize",
            "token_endpoint": "https://api.ibkr.com/oauth2/api/v1/token",
            "registration_endpoint": "https://api.ibkr.com/oauth2/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
        }).encode(),
    )

    register_req = await flow.asend(asm_resp)
    assert str(register_req.url) == "https://api.ibkr.com/oauth2/register"
    assert register_req.headers.get("User-Agent") == "Hermes-Agent"
    assert register_req.headers.get("Content-Type") == "application/json"
    body = json.loads(register_req.content.decode())
    assert body["scope"] == "mcp.read"
    assert provider.context.client_metadata.scope == "mcp.read"
    assert body["token_endpoint_auth_method"] == "none"

    bad_register = httpx.Response(
        400,
        request=register_req,
        text="forced test stop",
    )
    with pytest.raises(Exception):
        await flow.asend(bad_register)
