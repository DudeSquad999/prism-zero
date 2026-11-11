"""SVG safety and brand-identity regressions; standard library plus pytest."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import pytest

ROOT = Path(__file__).resolve().parents[1]
NS = {'svg': 'http://www.w3.org/2000/svg'}

@pytest.mark.parametrize('name', ['prism-mark.svg', 'hero.svg', 'architecture.svg'])
def test_svg_is_self_contained(name: str) -> None:
    text = (ROOT / 'assets' / name).read_text(encoding='utf-8')
    assert '<!DOCTYPE' not in text and '<!ENTITY' not in text
    root = ET.fromstring(text)
    assert root.tag == '{http://www.w3.org/2000/svg}svg'
    assert root.get('role') == 'img'
    bounds = [float(value) for value in root.attrib['viewBox'].split()]
    assert len(bounds) == 4 and bounds[2] > 0 and bounds[3] > 0
    ids = [element.attrib['id'] for element in root.iter() if 'id' in element.attrib]
    assert len(ids) == len(set(ids))
    for element in root.iter():
        assert element.tag.rsplit('}', 1)[-1] not in {'script', 'foreignObject', 'image', 'style'}
        for attribute, value in element.attrib.items():
            local = attribute.rsplit('}', 1)[-1].lower()
            assert not local.startswith('on')
            if local in {'href', 'src'}:
                assert value.startswith('#') and value[1:] in ids
            for reference in re.findall(r'url\(([^)]+)\)', value):
                assert reference.startswith('#') and reference[1:] in ids
    for label in root.get('aria-labelledby', '').split():
        assert label in ids

@pytest.mark.parametrize('name', ['hero.svg', 'architecture.svg'])
def test_diagrams_have_accessible_descriptions(name: str) -> None:
    root = ET.parse(ROOT / 'assets' / name).getroot()
    assert root.find('svg:title', NS).text
    assert root.find('svg:desc', NS).text
    assert root.get('aria-labelledby') == 'title desc'

def test_hero_preserves_supplied_mark_geometry_and_colors() -> None:
    original = ET.parse(ROOT / 'assets' / 'prism-mark.svg').getroot()
    hero = ET.parse(ROOT / 'assets' / 'hero.svg').getroot()
    for class_name in ['body', 'ray']:
        reference = original.findall(f".//svg:path[@class='{class_name}']", NS)
        actual = hero.findall(f".//svg:path[@class='{class_name}']", NS)
        assert [node.attrib for node in actual] == [node.attrib for node in reference]
    original_stops = original.findall('.//svg:linearGradient/svg:stop', NS)
    hero_stops = hero.findall(".//svg:linearGradient[@id='prism-face']/svg:stop", NS)
    assert [node.attrib for node in hero_stops] == [node.attrib for node in original_stops]
