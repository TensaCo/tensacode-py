"""Recorded labels remain supplied DOM evidence, independent of pixel heuristics."""
import sys

from examples.browser_agents.perception.protocol import Target
from examples.browser_agents.perception.web_dom import RecordedDomProvider


def test_recorded_dom_transports_supplied_labels_without_visual_evaluation(monkeypatch):
    monkeypatch.setitem(sys.modules, 'eval.vision_perception', None)
    controls = [dict(role='button', name=name, value='', hint='', section='',
                     box=[10, 10, 20, 20], point=[15, 15])
                for name in ('ARIA-only supplied label', '')]
    screen = dict(controls=controls, texts=[], graphics=[], url='https://example.test',
                  title='Captured document', dialog='')
    perceived = RecordedDomProvider(screen).perceive(Target())
    assert [element.name for element in perceived.elements] == ['ARIA-only supplied label', '']
    assert all(element.sources == ('web-dom',) for element in perceived.elements)
    assert [control['name'] for control in screen['controls']] == ['ARIA-only supplied label', '']
    assert perceived.url == screen['url']
