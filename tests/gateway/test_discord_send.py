from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock
import sys

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_system_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    ref_msg = SimpleNamespace(id=99)
    sent_msg = SimpleNamespace(id=1234)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return sent_msg

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "1234"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 2
    assert send_calls[0]["reference"] is ref_msg
    assert send_calls[1]["reference"] is None


@pytest.mark.asyncio
async def test_send_voice_mp3_uses_regular_attachment_and_preserves_reply(tmp_path, monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake mp3 bytes")

    ref_msg = SimpleNamespace(id=777)
    sent_msg = SimpleNamespace(id=4321)
    discord_file = object()

    existing_http = getattr(getattr(discord_platform, "discord", None), "http", SimpleNamespace(Route=MagicMock()))
    monkeypatch.setattr(
        discord_platform,
        "discord",
        SimpleNamespace(File=lambda *args, **kwargs: discord_file, http=existing_http),
    )

    channel = SimpleNamespace(
        id=55,
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(return_value=sent_msg),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=AsyncMock(side_effect=AssertionError("raw voice API should not be used for mp3"))),
    )

    result = await adapter.send_voice("55", str(audio_path), caption="escuchá esto", reply_to="777")

    assert result.success is True
    assert result.message_id == "4321"
    channel.fetch_message.assert_awaited_once_with(777)
    channel.send.assert_awaited_once_with(content="escuchá esto", file=discord_file, reference=ref_msg)
    assert result.raw_response == {"content_type": "audio/mpeg"}


@pytest.mark.asyncio
async def test_send_voice_ogg_uses_native_voice_api_without_attachment_fallback(tmp_path, monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"OggSfake")

    existing_file = getattr(getattr(discord_platform, "discord", None), "File", MagicMock())
    monkeypatch.setattr(
        discord_platform,
        "discord",
        SimpleNamespace(File=existing_file, http=SimpleNamespace(Route=MagicMock(return_value="route"))),
    )

    channel = SimpleNamespace(
        id=66,
        send=AsyncMock(side_effect=AssertionError("fallback attachment should not be used when raw voice send succeeds")),
    )
    http_client = SimpleNamespace(request=AsyncMock(return_value={"id": "999"}))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=http_client,
    )

    result = await adapter.send_voice("66", str(audio_path))

    assert result.success is True
    assert result.message_id == "999"
    http_client.request.assert_awaited_once_with("route", form=ANY)
    channel.send.assert_not_called()
