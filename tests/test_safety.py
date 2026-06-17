import pytest
from databridge.safety.enforcement import SafetyEnforcer, SafetyViolation, ViolationType


@pytest.fixture
def enforcer():
    return SafetyEnforcer()


def test_select_passes(enforcer):
    enforcer.check("SELECT id, name FROM customers WHERE id = 1")


def test_select_with_join_passes(enforcer):
    enforcer.check("SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id")


def test_insert_blocked(enforcer):
    with pytest.raises(SafetyViolation) as exc:
        enforcer.check("INSERT INTO orders VALUES (1, 2, 100.0)")
    assert exc.value.violation_type == ViolationType.WRITE_OPERATION


def test_update_blocked(enforcer):
    with pytest.raises(SafetyViolation) as exc:
        enforcer.check("UPDATE orders SET amount = 0 WHERE id = 1")
    assert exc.value.violation_type == ViolationType.WRITE_OPERATION


def test_delete_blocked(enforcer):
    with pytest.raises(SafetyViolation) as exc:
        enforcer.check("DELETE FROM orders")
    assert exc.value.violation_type == ViolationType.WRITE_OPERATION


def test_drop_blocked(enforcer):
    with pytest.raises(SafetyViolation) as exc:
        enforcer.check("DROP TABLE orders")
    assert exc.value.violation_type == ViolationType.DDL_OPERATION


def test_create_blocked(enforcer):
    with pytest.raises(SafetyViolation) as exc:
        enforcer.check("CREATE TABLE foo (id INTEGER)")
    assert exc.value.violation_type == ViolationType.DDL_OPERATION


def test_multiple_statements_blocked(enforcer):
    with pytest.raises(SafetyViolation) as exc:
        enforcer.check("SELECT 1; SELECT 2")
    assert exc.value.violation_type == ViolationType.MULTIPLE_STATEMENTS


def test_mongo_pipeline_passes(enforcer):
    # JSON strings are always read-only — skip SQL parsing
    enforcer.check('{"collection": "orders", "pipeline": [{"$match": {"status": "shipped"}}]}')


def test_empty_query_passes(enforcer):
    enforcer.check("")
    enforcer.check("   ")


def test_is_safe_true(enforcer):
    assert enforcer.is_safe("SELECT 1") is True


def test_is_safe_false(enforcer):
    assert enforcer.is_safe("DROP TABLE users") is False
