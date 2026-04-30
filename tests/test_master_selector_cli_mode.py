from alpha_council.master_selector import skip_if_no_analysis_intent


class DummyContext:
    def __init__(self, state: dict):
        self.state = state


def test_master_selector_skips_when_cli_flag_enabled() -> None:
    result = skip_if_no_analysis_intent(DummyContext({"skip_master_selector": True}))
    assert result is not None
    assert result.parts == []
