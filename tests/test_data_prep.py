from src.data_prep import dedup, stratified_split, to_example, validate_all
from src.schema import parse_and_validate


def _mk(sent, i):
    return to_example(f"text {sent} {i}", sent, ["E"], "summary line.")


def test_to_example_is_schema_valid():
    e = _mk("positive", 0)
    assert parse_and_validate(e["output"]).ok
    assert e["sentiment"] == "positive"


def test_validate_drops_bad():
    good = _mk("positive", 0)
    bad = dict(good, output='{"sentiment":"x"}')
    assert len(validate_all([good, bad])) == 1


def test_dedup():
    a = _mk("neutral", 0)
    b = dict(a)  # same input
    assert len(dedup([a, b])) == 1


def test_split_is_stratified_and_seeded():
    rows = [_mk("positive", i) for i in range(10)]
    rows += [_mk("negative", i) for i in range(10)]
    rows += [_mk("neutral", i) for i in range(10)]
    s1 = stratified_split(rows)
    s2 = stratified_split(rows)
    # seeded -> deterministic
    assert [r["input"] for r in s1["train"]] == [r["input"] for r in s2["train"]]
    # no leakage across splits
    inputs = lambda k: {r["input"] for r in s1[k]}
    assert inputs("train") & inputs("test") == set()
    assert inputs("train") & inputs("val") == set()
    # per class of 10: train 8 / val 1 / test 1  -> totals 24/3/3
    assert len(s1["train"]) == 24
    assert len(s1["val"]) == 3 and len(s1["test"]) == 3
