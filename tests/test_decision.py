import pytest

from cavity_encoding_ml import decide


def test_use_cavity():
    d = decide(0.525, 0.09)
    assert d.verdict == "use_cavity" and d.ratio == pytest.approx(0.525 / 0.09)


def test_adds_noise_like_tensile():
    d = decide(0.011, 0.037)
    assert d.verdict == "adds_noise"
    assert "adds noise" in d.message and "leave the cavity label out" in d.message


def test_similar_jitter_counts_as_noise():
    assert decide(0.05, 0.05).verdict == "adds_noise"


def test_exactly_at_margin_is_kept():
    assert decide(0.10, 0.05).verdict == "use_cavity"


def test_custom_margin():
    assert decide(0.10, 0.05, margin=3).verdict == "adds_noise"


def test_failed_diagnostic_wins():
    d = decide(0.9, 0.01, diagnostic_passed=False)
    assert d.verdict == "no_cavity_effect"


def test_zero_jitter():
    assert decide(0.1, 0.0).verdict == "use_cavity"


def test_report_and_dict():
    d = decide(0.3, 0.05)
    assert d.report().startswith("STEP 4") and d.to_dict()["verdict"] == "use_cavity"
