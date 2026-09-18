"""The provider protocol, fusion rules, and provider availability (no models, no browser)."""

import pytest

from examples.browser_agents.perception.ax import AxProvider
from examples.browser_agents.perception.fixture import FixtureProvider
from examples.browser_agents.perception.fusion import FusedProvider, fuse_scenes
from examples.browser_agents.perception.protocol import Element, PerceivedScene, Target, TextBlock, own
from examples.browser_agents.perception.uia import UiaProvider


def scene(*, name="", text="", conf=0.9, source="web-dom", box=(10, 10, 60, 20)):
    return PerceivedScene(
        elements=(Element("button", name, box=box, point=(box[0] + 5, box[1] + 5), confidence=conf, provenance=own(source, "m")),),
        texts=(TextBlock(text, (10, 50, 80, 14), confidence=conf, provenance=own(source, "ocr")),) if text else (),
        source=source,
    )


def fused(a, b, ra=0.95, rb=0.6):
    return FusedProvider([FixtureProvider(a, "web-dom", ra), FixtureProvider(b, "vision", rb)]).perceive(Target())


def test_a_missing_name_is_filled_from_the_weaker_provider_with_attribution():
    out = fused(scene(name=""), scene(name="Save", conf=0.7, source="vision"))
    e = out.elements[0]
    assert e.name == "Save" and e.sources == ("web-dom", "vision") and not e.conflicts


def test_agreement_raises_confidence_and_keeps_both_sources():
    out = fused(scene(name="Save", conf=0.9), scene(name="Save", conf=0.7, source="vision"))
    e = out.elements[0]
    assert e.name == "Save" and e.confidence > 0.9 and e.sources == ("web-dom", "vision")


def test_disagreement_keeps_the_more_reliable_reading_but_lowers_confidence_and_records_it():
    out = fused(scene(name="Save"), scene(name="Send", conf=0.9, source="vision"))
    e = out.elements[0]
    assert e.name == "Save" and e.confidence < 0.9
    assert [(c.field, c.mine, c.theirs, c.source) for c in e.conflicts] == [("name", "Save", "Send", "vision")]
    assert e in out.uncertain


def test_equally_reliable_disagreement_becomes_unknown_rather_than_a_guess():
    out = fuse_scenes(scene(text="Total: $5"), scene(text="Total: $8", source="vision"), 0.7, 0.7)
    t = out.texts[0]
    assert t.text == "" and t.confidence == 0.0 and t.conflicts


def test_elements_only_one_provider_saw_are_kept():
    only_vision = PerceivedScene(elements=(Element("button", "Zoom", box=(500, 500, 40, 20), point=(520, 510), confidence=0.5, provenance=own("vision", "ocr")),), source="vision")
    out = fused(scene(name="Save"), only_vision)
    assert sorted(e.name for e in out.elements) == ["Save", "Zoom"]


def test_scene_flattens_to_the_screen_agents_already_use():
    out = fused(scene(name=""), scene(name="Save", conf=0.7, source="vision"))
    screen = out.to_screen()
    assert screen.controls[0].name == "Save" and screen.controls[0].role == "button"


def test_platform_providers_report_availability_instead_of_pretending():
    for provider in (UiaProvider(), AxProvider()):
        assert provider.available() is False
        with pytest.raises(NotImplementedError):
            provider.perceive(Target())


def test_atspi_provider_is_usable_on_this_machine_or_says_why():
    atspi = pytest.importorskip("gi")  # noqa: F841
    from examples.browser_agents.perception.atspi import AtspiProvider

    provider = AtspiProvider()
    if not provider.available():
        pytest.skip("no AT-SPI bus in this environment")
    windows = provider.windows()
    assert isinstance(windows, list)


def test_a_partial_view_inside_a_control_is_merged_not_added_as_a_rival():
    """Vision often sees only a checkbox's label; clicking that instead of the widget misses."""
    widget = PerceivedScene(elements=(Element("checkbox", "Urgent", box=(100, 100, 74, 34), point=(137, 117), confidence=0.95, provenance=own("atspi", "role=check box")),), source="atspi")
    label_only = PerceivedScene(elements=(Element("button", "Urgent", box=(122, 108, 44, 16), point=(144, 116), confidence=0.6, provenance=own("vision", "ocr")),), source="vision")
    out = FusedProvider([FixtureProvider(widget, "atspi", 0.93), FixtureProvider(label_only, "vision", 0.6)]).perceive(Target())
    assert len(out.elements) == 1
    e = out.elements[0]
    assert e.role == "checkbox" and e.point == (137, 117) and e.sources == ("atspi", "vision")
