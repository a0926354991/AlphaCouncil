import asyncio

from alpha_council.cli import run


def test_parse_masters_by_index_and_name() -> None:
    selected = run.parse_masters("1, ben_graham,1, peter-lynch")
    assert selected == ["warren_buffett", "ben_graham", "peter_lynch"]


def test_parse_masters_empty_is_skip() -> None:
    assert run.parse_masters("") == []
    assert run.parse_masters(None) == []


def test_parse_masters_invalid_raises() -> None:
    try:
        run.parse_masters("999")
    except run.CliUsageError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected CliUsageError")


def test_infer_market_and_ticker_normalization() -> None:
    assert run._infer_market("2330", None) == "tw"
    assert run._infer_market("aapl", None) == "us"
    assert run._normalize_ticker("aapl", "us") == "AAPL"
    assert run._normalize_ticker("2330", "tw") == "2330"


def test_build_user_message_defaults_to_skip_when_no_masters() -> None:
    msg = run._build_user_message("2330", "tw", None)
    assert "2330" in msg
    assert "master_choice=0" in msg


def test_build_user_message_includes_masters_choice() -> None:
    msg = run._build_user_message("AAPL", "us", "1,2,3")
    assert "AAPL US" in msg
    assert "master_choice=1,2,3" in msg


def test_run_command_persist_enabled(monkeypatch) -> None:
    outcome = run.RunOutcome(
        status="SUCCEEDED", final_text="buy", session_id="s1", state={}, event_count=3
    )

    async def fake_run_pipeline(**kwargs):
        return outcome

    monkeypatch.setattr(run, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run, "_has_model_auth", lambda: True)
    monkeypatch.setattr(run, "_persist_report", lambda *args, **kwargs: "./reports/x.json")
    monkeypatch.setenv("ALPHACOUNCIL_PERSIST_ENABLED", "true")

    args = run.build_parser().parse_args(["run", "--ticker", "2330", "--market", "tw"])
    code = asyncio.run(run._run_command(args))
    assert code == run.EXIT_OK


def test_run_command_failed_status(monkeypatch) -> None:
    outcome = run.RunOutcome(
        status="FAILED", final_text="", session_id="s1", state={}, event_count=2
    )

    async def fake_run_pipeline(**kwargs):
        return outcome

    monkeypatch.setattr(run, "_run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(run, "_has_model_auth", lambda: True)
    args = run.build_parser().parse_args(["run", "--ticker", "AAPL", "--market", "us"])
    code = asyncio.run(run._run_command(args))
    assert code == run.EXIT_EXPECTED_ERROR
