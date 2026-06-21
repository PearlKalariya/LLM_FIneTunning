from src.schema import parse_and_validate


def test_valid():
    out = '{"sentiment":"positive","key_entities":["TCS"],"summary":"TCS up."}'
    r = parse_and_validate(out)
    assert r.ok and r.data["sentiment"] == "positive"


def test_strips_code_fence():
    out = '```json\n{"sentiment":"neutral","key_entities":[],"summary":"flat."}\n```'
    assert parse_and_validate(out).ok


def test_bad_sentiment():
    out = '{"sentiment":"bullish","key_entities":[],"summary":"x."}'
    r = parse_and_validate(out)
    assert not r.ok and "bad sentiment" in r.error


def test_missing_key():
    out = '{"sentiment":"neutral","summary":"x."}'
    r = parse_and_validate(out)
    assert not r.ok and "missing keys" in r.error


def test_bad_json():
    assert not parse_and_validate("not json").ok


def test_entities_must_be_str_list():
    out = '{"sentiment":"neutral","key_entities":[1,2],"summary":"x."}'
    assert not parse_and_validate(out).ok


def test_empty_summary():
    out = '{"sentiment":"neutral","key_entities":[],"summary":"  "}'
    assert not parse_and_validate(out).ok
