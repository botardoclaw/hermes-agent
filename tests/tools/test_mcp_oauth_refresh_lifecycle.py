"""Offline lifecycle tracing: real SDK/fence/storage, exclusively synthetic secrets."""
import asyncio
import json
import logging
import os
import re
from unittest.mock import AsyncMock

import httpx2
import pytest

from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_manager import HermesMCPOAuthProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthMetadata, OAuthToken

PREFIX = "MCP OAuth refresh lifecycle "
SECRETS = ("ACCESS-SENTINEL", "REFRESH-SENTINEL", "CLIENT-SENTINEL", "SECRET-SENTINEL")


async def provider_for(tmp_path, name="ibkr"):
    storage = HermesTokenStorage(name, hermes_home=tmp_path)
    provider = HermesMCPOAuthProvider(
        server_url="https://resource.example/mcp", storage=storage,
        client_metadata=OAuthClientMetadata(redirect_uris=["http://localhost/cb"]),
        redirect_handler=AsyncMock(), callback_handler=AsyncMock(),
    )
    provider.context.client_info = OAuthClientInformationFull(
        client_id=SECRETS[2], client_secret=SECRETS[3],
        token_endpoint_auth_method="client_secret_post", redirect_uris=["http://localhost/cb"],
        issuer="https://issuer.example/",
    )
    provider.context.oauth_metadata = OAuthMetadata(
        issuer="https://issuer.example", authorization_endpoint="https://issuer.example/authorize",
        token_endpoint="https://issuer.example/token",
    )
    provider.context.current_tokens = OAuthToken(
        access_token=SECRETS[0], refresh_token=SECRETS[1], token_type="Bearer", expires_in=0,
    )
    await storage.set_client_info(provider.context.client_info)
    storage.bind_issuer("https://issuer.example/")
    await storage.set_tokens(provider.context.current_tokens)
    return provider


def records(caplog):
    rows = [json.loads(r.getMessage().split(PREFIX, 1)[1]) for r in caplog.records if PREFIX in r.getMessage()]
    # Inspect args too: deferred formatting/structured log collectors must be safe.
    text = caplog.text + repr([(r.msg, r.args) for r in caplog.records])
    import hashlib
    for value in SECRETS:
        assert value not in text
        assert hashlib.sha256(value.encode()).hexdigest() not in text
    for value in ("https://issuer.example", "https://resource.example", "description-sentinel"):
        assert value not in text
    for row in rows:
        assert row["server"] == "ibkr"
        assert row["pid"] == os.getpid()
        assert re.fullmatch(r"[0-9a-f]{32}", row["attempt"])
        assert all(type(v) is bool for k, v in row.items() if k not in {"server", "pid", "attempt", "event", "status", "oauth_error"})
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", [False, True])
async def test_success_lifecycle_and_new_random_attempt(tmp_path, caplog, resource):
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    provider.context.protocol_version = "2025-06-18" if resource else None
    ids = []
    for _ in range(2):
        await provider._refresh_token()
        assert await provider._handle_refresh_response(httpx2.Response(200, json={
            "access_token": SECRETS[0], "refresh_token": SECRETS[1], "token_type": "Bearer",
        }))
        assert provider._hermes_fence is None
        rows = records(caplog)
        ids.append(rows[-1]["attempt"])
    assert ids[0] != ids[1]
    first = [r for r in rows if r["attempt"] == ids[0]]
    assert [r["event"] for r in first] == [
        "attempt_started", "lock_acquired", "disk_checked", "request_ready",
        "response", "persisted", "lock_released", "attempt_finished",
    ]
    assert first[2]["access_changed"] is False and first[2]["refresh_changed"] is False
    assert first[2]["issuer_binding_match"] is True
    request = first[3]
    assert request["client_binding_match"] is True
    assert request["client_issuer_match"] is True
    assert request["resource_present"] is resource and request["resource_match"] is resource
    assert first[5]["persistence_complete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("client_present,match", [(True, False), (False, False)])
async def test_binding_difference_or_unavailable_is_boolean(tmp_path, caplog, client_present, match):
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    path = provider.context.storage._client_info_path()
    if client_present:
        path.write_text(json.dumps({"client_id": "OTHER-SENTINEL", "issuer": "https://other.example/"}))
    else:
        path.unlink()
    try:
        await provider._refresh_token()
        row = next(r for r in records(caplog) if r["event"] == "request_ready")
        assert row["client_binding_available"] is client_present
        assert row["client_binding_match"] is match
        assert row["client_issuer_match"] is False
        assert "OTHER-SENTINEL" not in caplog.text
    finally:
        await provider._hermes_release_refresh_fence()


@pytest.mark.asyncio
async def test_rejection_and_same_refresh_changed_access(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    disk = provider.context.current_tokens.model_copy(update={"access_token": "OTHER-SENTINEL"})
    await provider.context.storage.set_tokens(disk)
    await provider._refresh_token()
    assert await provider._handle_refresh_response(httpx2.Response(400, json={
        "error": "invalid_grant", "error_description": "description-sentinel", "refresh_token": SECRETS[1],
    })) is False
    rows = records(caplog)
    checked = next(r for r in rows if r["event"] == "disk_checked")
    assert checked["access_changed"] is True and checked["refresh_changed"] is False
    assert next(r for r in rows if r["event"] == "response")["oauth_error"] == "invalid_grant"
    assert "persisted" not in [r["event"] for r in rows]
    assert [r["event"] for r in rows][-3:] == ["rejected_state_cleared", "lock_released", "attempt_finished"]
    assert await provider.context.storage.get_tokens() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("description-sentinel"), asyncio.CancelledError()])
async def test_persistence_failure_never_claims_success_and_releases(tmp_path, caplog, failure):
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    await provider._refresh_token()
    provider.context.storage.set_tokens = AsyncMock(side_effect=failure)
    with pytest.raises(type(failure)):
        await provider._handle_refresh_response(httpx2.Response(200, json={"access_token": SECRETS[0], "token_type": "Bearer"}))
    rows = records(caplog)
    assert next(r for r in rows if r["event"] == "persisted")["persistence_complete"] is False
    assert rows[-2]["event"] == "lock_released"
    assert provider._hermes_fence is None


@pytest.mark.asyncio
async def test_lock_failure_no_post_and_correlation_terminated(tmp_path, caplog, monkeypatch):
    from tools.mcp_oauth import RefreshFenceTimeout
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    monkeypatch.setattr("tools.mcp_oauth.acquire_refresh_fence", AsyncMock(side_effect=RefreshFenceTimeout("description-sentinel")))
    with pytest.raises(RefreshFenceTimeout):
        await provider._refresh_token()
    assert [r["event"] for r in records(caplog)] == ["attempt_started", "lock_not_acquired", "attempt_finished"]
    assert provider._hermes_refresh_attempt is None


@pytest.mark.asyncio
async def test_unrelated_server_and_broken_logger_do_not_affect_refresh(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path, "not-ibkr")
    await provider._refresh_token()
    await provider._hermes_release_refresh_fence()
    assert records(caplog) == []
    provider = await provider_for(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("description-sentinel")
    monkeypatch.setattr(provider._hermes_logger, "info", broken)
    await provider._refresh_token()
    assert await provider._handle_refresh_response(httpx2.Response(200, json={"access_token": SECRETS[0], "token_type": "Bearer"}))
    assert provider._hermes_fence is None


@pytest.mark.asyncio
async def test_abandoned_generator_releases_correlated_lock(tmp_path, caplog):
    import time
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    provider._initialized = True
    provider.context.token_expiry_time = time.time() - 100
    from tools.mcp_oauth_provider import HermesProviderMixin
    flow = HermesProviderMixin.async_auth_flow(provider, httpx2.Request("GET", "https://resource.example/mcp"))
    outbound = await flow.asend(None)
    assert outbound.method == "POST"
    await flow.aclose()
    rows = records(caplog)
    assert rows[-2]["event"] == "lock_released"
    assert rows[-1]["event"] == "attempt_finished"
    assert provider._hermes_fence is None
    assert not any(r["event"] == "persisted" for r in rows)


@pytest.mark.asyncio
async def test_live_rotated_peer_skips_post_without_changing_semantics(tmp_path, caplog):
    from tools.mcp_oauth_provider import _RefreshCompletedByPeer
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    disk = provider.context.current_tokens.model_copy(update={
        "access_token": "PEER-ACCESS", "refresh_token": "PEER-REFRESH", "expires_in": 3600,
    })
    await provider.context.storage.set_tokens(disk)
    with pytest.raises(_RefreshCompletedByPeer):
        await provider._refresh_token()
    rows = records(caplog)
    checked = next(r for r in rows if r["event"] == "disk_checked")
    assert checked["access_changed"] is True and checked["refresh_changed"] is True
    assert not any(r["event"] == "request_ready" for r in rows)
    assert rows[-2]["event"] == "lock_released"
    assert "PEER-ACCESS" not in caplog.text and "PEER-REFRESH" not in caplog.text


@pytest.mark.asyncio
async def test_trace_schema_rejects_arbitrary_text_and_does_not_write_registration(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    provider = await provider_for(tmp_path)
    path = provider.context.storage._client_info_path()
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    try:
        await provider._refresh_token()
        provider._hermes_trace_refresh("response", oauth_error="description-sentinel", status="description-sentinel",
                                      access_changed="description-sentinel", token=SECRETS[0])
        provider._hermes_trace_refresh("description-sentinel")
        rows = records(caplog)
        assert rows[-1]["event"] == "response"
        assert set(rows[-1]) == {"server", "pid", "attempt", "event"}
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    finally:
        await provider._hermes_release_refresh_fence()
