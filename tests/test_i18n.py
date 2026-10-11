"""UI i18n: EN/UZ/RU dictionaries are complete and consistent, and every key used by the pages exists."""
import json
import re
from pathlib import Path

import pytest

from pulse.app.main import STATIC

STATIC_DIR = Path(STATIC)
LANGS = ("en", "uz", "ru")
PLACEHOLDER = re.compile(r"\{(\w+)\}")
ENTRY = re.compile(r'^\s*("(?:[^"\\]|\\.)*"):\s*("(?:[^"\\]|\\.)*"),\s*$')


def _dicts() -> dict[str, dict[str, str]]:
    src = (STATIC_DIR / "i18n.js").read_text(encoding="utf-8")
    out = {}
    for lang in LANGS:
        m = re.search(rf"^const {lang} = \{{\n(.*?)^\}};", src, re.S | re.M)
        assert m, f"dictionary {lang} not found"
        d = {}
        for line in m.group(1).splitlines():
            if not line.strip():
                continue
            e = ENTRY.match(line)
            assert e, f"{lang}: unparseable line {line!r}"
            k, v = json.loads(e.group(1)), json.loads(e.group(2))
            assert k not in d, f"{lang}: duplicate key {k}"
            d[k] = v
        out[lang] = d
    return out


@pytest.fixture(scope="module")
def dicts():
    return _dicts()


def test_dictionaries_not_trivial(dicts):
    assert len(dicts["en"]) > 100


@pytest.mark.parametrize("lang", ["uz", "ru"])
def test_every_en_key_translated(dicts, lang):
    en, other = dicts["en"], dicts[lang]
    assert not set(en) - set(other), f"{lang} lacks {sorted(set(en) - set(other))}"
    assert not set(other) - set(en), f"{lang} has extra {sorted(set(other) - set(en))}"
    for k, v in other.items():
        assert v.strip(), f"{lang}.{k} is empty"


@pytest.mark.parametrize("lang", LANGS)
def test_non_empty_values(dicts, lang):
    for k, v in dicts[lang].items():
        assert v.strip(), f"{lang}.{k} is empty"


@pytest.mark.parametrize("lang", ["uz", "ru"])
def test_placeholders_match(dicts, lang):
    for k, v in dicts["en"].items():
        assert sorted(PLACEHOLDER.findall(v)) == sorted(PLACEHOLDER.findall(dicts[lang][k])), f"{lang}.{k}"


def test_scripts_are_not_mixed(dicts):
    cyr = re.compile(r"[Ѐ-ӿ]")
    for k, v in dicts["uz"].items():
        assert not cyr.search(v), f"uz.{k} must be Latin script"
    for k, v in dicts["ru"].items():
        if k not in {"page.title"}:
            plain = PLACEHOLDER.sub("", v)
            assert cyr.search(v) or not re.search(r"[A-Za-z]{4,}", plain), f"ru.{k} looks untranslated"


def _used_keys() -> set[str]:
    keys: set[str] = set()
    for p in STATIC_DIR.glob("*.html"):
        text = p.read_text(encoding="utf-8")
        keys |= set(re.findall(r'data-i18n="([^"]+)"', text))
        for attr in re.findall(r'data-i18n-attr="([^"]+)"', text):
            keys |= {pair.split(":", 1)[1].strip() for pair in attr.split(";") if ":" in pair}
    for p in STATIC_DIR.glob("*.js"):
        if p.name == "i18n.js":
            continue
        text = p.read_text(encoding="utf-8")
        keys |= set(re.findall(r'(?<![\w.$])t\(\s*"([^"]+)"', text))
    return keys


def test_used_keys_exist_in_en(dicts):
    used = _used_keys()
    assert len(used) > 80
    missing = used - set(dicts["en"])
    assert not missing, f"keys used but missing from en: {sorted(missing)}"


def test_t_calls_use_literal_keys():
    for p in STATIC_DIR.glob("*.js"):
        if p.name == "i18n.js":
            continue
        for m in re.finditer(r'(?<![\w.$])t\(\s*(.)', p.read_text(encoding="utf-8")):
            assert m.group(1) == '"', f"{p.name}: t() must take a string-literal key"


def test_no_unused_keys(dicts):
    unused = set(dicts["en"]) - _used_keys() - {"lang.aria"}
    assert not unused, f"dictionary keys never used: {sorted(unused)}"


def test_pages_have_switcher_and_english_banner():
    for name in ("index.html", "live.html"):
        html = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert 'id="lang-switch"' in html
        assert "Anonymous, aggregated analytics; decision support only." in html
        assert '/static/i18n.js' in (STATIC_DIR / ("app.js" if name == "index.html" else "live.js")).read_text()


def test_no_innerhtml_in_ui_scripts():
    for p in STATIC_DIR.glob("*.js"):
        assert "innerHTML" not in p.read_text(encoding="utf-8"), p.name


def test_median_is_translated_as_median(dicts):
    hits = [k for k, v in dicts["en"].items() if "median" in v.lower()]
    assert hits
    for k in hits:
        assert "median" in dicts["uz"][k].lower(), f"uz.{k} must say Mediana, not average"
        assert "медиан" in dicts["ru"][k].lower(), f"ru.{k} must say медиан-"


def test_unknown_stored_language_is_ignored():
    src = (STATIC_DIR / "i18n.js").read_text(encoding="utf-8")
    assert "Object.hasOwn(DICTS, saved)" in src
