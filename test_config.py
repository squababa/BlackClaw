import importlib
import sys


def _reload_config(monkeypatch, **env_overrides):
    for var in (
        "BLACKCLAW_CYCLE_COOLDOWN",
        "BLACKCLAW_COOLDOWN",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "LLM_PROVIDER",
        "LOCAL_LLM_ONLY",
        "BLACKCLAW_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    for var, value in env_overrides.items():
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, str(value))

    sys.modules.pop("config", None)
    import config

    return importlib.reload(config)


def test_cycle_cooldown_prefers_runtime_override_and_preserves_legacy_fallback(
    monkeypatch,
) -> None:
    config = _reload_config(monkeypatch)
    assert config.CYCLE_COOLDOWN == 300

    config = _reload_config(monkeypatch, BLACKCLAW_COOLDOWN="45")
    assert config.CYCLE_COOLDOWN == 45

    config = _reload_config(
        monkeypatch,
        BLACKCLAW_COOLDOWN="45",
        BLACKCLAW_CYCLE_COOLDOWN="10",
    )
    assert config.CYCLE_COOLDOWN == 10


def test_cycle_cooldown_invalid_override_falls_back_to_default_with_warning(
    monkeypatch,
    capsys,
) -> None:
    config = _reload_config(
        monkeypatch,
        BLACKCLAW_COOLDOWN="45",
        BLACKCLAW_CYCLE_COOLDOWN="invalid",
    )

    assert config.CYCLE_COOLDOWN == 300

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "BLACKCLAW_CYCLE_COOLDOWN" in captured.out
    assert "default 300" in captured.out


def test_valid_claude_provider_model_pair_loads(monkeypatch) -> None:
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="claude",
        BLACKCLAW_MODEL="claude-sonnet-4-6",
    )
    assert config.LLM_PROVIDER == "claude"
    assert config.MODEL == "claude-sonnet-4-6"


def test_valid_gemini_provider_model_pair_loads(monkeypatch) -> None:
    config = _reload_config(
        monkeypatch,
        LLM_PROVIDER="gemini",
        BLACKCLAW_MODEL="gemini-2.5-flash",
    )
    assert config.LLM_PROVIDER == "gemini"
    assert config.MODEL == "gemini-2.5-flash"


def test_invalid_gemini_provider_with_claude_model_fails_fast(
    monkeypatch,
    capsys,
) -> None:
    try:
        _reload_config(
            monkeypatch,
            LLM_PROVIDER="gemini",
            BLACKCLAW_MODEL="claude-sonnet-4-6",
        )
        raise AssertionError("Expected config import to fail for invalid provider/model pair")
    except SystemExit as exc:
        assert exc.code == 1

    captured = capsys.readouterr()
    assert "BLACKCLAW_MODEL='claude-sonnet-4-6'" in captured.out
    assert "LLM_PROVIDER='gemini'" in captured.out
    assert "not compatible" in captured.out


def test_invalid_claude_provider_with_gemini_model_fails_fast(
    monkeypatch,
    capsys,
) -> None:
    try:
        _reload_config(
            monkeypatch,
            LLM_PROVIDER="claude",
            BLACKCLAW_MODEL="gemini-2.5-flash",
        )
        raise AssertionError("Expected config import to fail for invalid provider/model pair")
    except SystemExit as exc:
        assert exc.code == 1

    captured = capsys.readouterr()
    assert "BLACKCLAW_MODEL='gemini-2.5-flash'" in captured.out
    assert "LLM_PROVIDER='claude'" in captured.out
    assert "not compatible" in captured.out
