import pytest
from databridge.schema.joins.transforms import TRANSFORM_GRAMMAR, detect_transform
from databridge.schema.joins.registry import JoinRegistry, JoinRule


# ── Transform grammar ──────────────────────────────────────────────────────────

def test_identity_transform():
    name, fn = next(t for t in TRANSFORM_GRAMMAR if t[0] == "identity")
    assert fn("  hello  ") == "hello"


def test_lowercase_transform():
    name, fn = next(t for t in TRANSFORM_GRAMMAR if t[0] == "lowercase")
    assert fn("Hello") == "hello"


def test_extract_digits_transform():
    name, fn = next(t for t in TRANSFORM_GRAMMAR if t[0] == "extract_digits")
    assert fn("ABC123") == "123"


def test_zfill5_transform():
    name, fn = next(t for t in TRANSFORM_GRAMMAR if t[0] == "zfill_5")
    assert fn("42") == "00042"


def test_remove_separators_transform():
    name, fn = next(t for t in TRANSFORM_GRAMMAR if t[0] == "remove_separators")
    assert fn("US-001") == "US001"


def test_detect_transform_identity_match():
    values_a = ["10001", "10002", "10003"]
    values_b = ["10001", "10002", "10003", "10004"]
    name, ratio = detect_transform(values_a, values_b)
    assert name == "identity"
    assert ratio >= 0.70


def test_detect_transform_zfill_match():
    values_a = [10001, 10002, 10003]
    values_b = ["10001", "10002", "10003", "10004"]
    name, ratio = detect_transform(values_a, values_b)
    assert name is not None
    assert ratio >= 0.70


def test_detect_transform_no_match():
    values_a = ["AAA", "BBB", "CCC"]
    values_b = ["111", "222", "333"]
    name, ratio = detect_transform(values_a, values_b)
    assert name is None


def test_detect_transform_empty_b():
    name, ratio = detect_transform(["a", "b"], [])
    assert name is None
    assert ratio == 0.0


def test_detect_transform_custom_threshold():
    values_a = ["1", "2", "3", "4", "5"]
    values_b = ["1", "2", "3"]  # 60% overlap
    name, ratio = detect_transform(values_a, values_b, overlap_threshold=0.5)
    assert name is not None


# ── JoinRegistry ──────────────────────────────────────────────────────────────

def _make_rule(join_id: str = "rule-1", confidence: float = 0.8) -> JoinRule:
    return JoinRule(
        join_id=join_id,
        db_a="db_a", table_a="orders", column_a="customer_id",
        db_b="db_b", table_b="customers", column_b="id",
        confidence=confidence,
        confirmed=True,
    )


@pytest.mark.asyncio
async def test_join_registry_save_and_get(join_registry):
    rule = _make_rule()
    await join_registry.save(rule)
    fetched = await join_registry.get("rule-1")
    assert fetched is not None
    assert fetched.join_id == "rule-1"
    assert fetched.confidence == pytest.approx(0.8)
    assert fetched.confirmed is True


@pytest.mark.asyncio
async def test_join_registry_get_missing(join_registry):
    result = await join_registry.get("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_join_registry_get_all(join_registry):
    await join_registry.save(_make_rule("r1", 0.9))
    await join_registry.save(_make_rule("r2", 0.7))
    all_rules = await join_registry.get_all()
    assert len(all_rules) == 2
    assert all_rules[0].confidence >= all_rules[1].confidence


@pytest.mark.asyncio
async def test_join_registry_confirmed_only(join_registry):
    confirmed = _make_rule("c1")
    unconfirmed = JoinRule(
        join_id="u1", db_a="x", table_a="t1", column_a="c1",
        db_b="y", table_b="t2", column_b="c2", confidence=0.5, confirmed=False,
    )
    await join_registry.save(confirmed)
    await join_registry.save(unconfirmed)
    only_confirmed = await join_registry.get_all(confirmed_only=True)
    assert all(r.confirmed for r in only_confirmed)
    assert len(only_confirmed) == 1


@pytest.mark.asyncio
async def test_join_registry_confirm(join_registry):
    rule = JoinRule(
        join_id="pending", db_a="a", table_a="t1", column_a="c1",
        db_b="b", table_b="t2", column_b="c2", confidence=0.65, confirmed=False,
    )
    await join_registry.save(rule)
    await join_registry.confirm("pending", session_id="test")
    updated = await join_registry.get("pending")
    assert updated.confirmed is True
    assert updated.verified_by == "test"


@pytest.mark.asyncio
async def test_join_registry_reject(join_registry):
    await join_registry.save(_make_rule("to-delete"))
    await join_registry.reject("to-delete")
    assert await join_registry.get("to-delete") is None


@pytest.mark.asyncio
async def test_join_registry_record_outcome_success(join_registry):
    rule = _make_rule("r-outcome", confidence=0.80)
    await join_registry.save(rule)
    await join_registry.record_outcome("r-outcome", success=True)
    updated = await join_registry.get("r-outcome")
    assert updated.confidence == pytest.approx(0.82)
    assert updated.success_count == 1


@pytest.mark.asyncio
async def test_join_registry_record_outcome_failure(join_registry):
    rule = _make_rule("r-fail", confidence=0.80)
    await join_registry.save(rule)
    await join_registry.record_outcome("r-fail", success=False)
    updated = await join_registry.get("r-fail")
    assert updated.confidence == pytest.approx(0.75)
    assert updated.failure_count == 1


@pytest.mark.asyncio
async def test_join_registry_find_for_tables(join_registry):
    rule = _make_rule("find-test")
    await join_registry.save(rule)
    results = await join_registry.find_for_tables("db_a", "orders", "db_b", "customers")
    assert len(results) == 1
    assert results[0].join_id == "find-test"


@pytest.mark.asyncio
async def test_join_rule_is_reliable_true():
    rule = _make_rule(confidence=0.75)
    assert rule.is_reliable is True


@pytest.mark.asyncio
async def test_join_rule_is_reliable_false_not_confirmed():
    rule = JoinRule(
        join_id="x", db_a="a", table_a="t", column_a="c",
        db_b="b", table_b="t2", column_b="c2",
        confidence=0.90, confirmed=False,
    )
    assert rule.is_reliable is False


@pytest.mark.asyncio
async def test_join_rule_is_reliable_false_low_confidence():
    rule = JoinRule(
        join_id="x", db_a="a", table_a="t", column_a="c",
        db_b="b", table_b="t2", column_b="c2",
        confidence=0.50, confirmed=True,
    )
    assert rule.is_reliable is False
