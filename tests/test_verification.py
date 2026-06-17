import pytest
from databridge.connectors.base import ColumnMeta, TableMeta
from databridge.query.executor import ExecutionResult
from databridge.verification.plausibility import (
    FailureMode,
    PlausibilityChecker,
    PlausibilityResult,
)


def _result(rows: list[dict]) -> ExecutionResult:
    cols = list(rows[0].keys()) if rows else []
    return ExecutionResult(rows=rows, row_count=len(rows), columns=cols)


def _schema(table_name: str = "orders", row_count: int = 0) -> dict:
    t = TableMeta(name=table_name, row_count_approx=row_count)
    t.columns["id"] = ColumnMeta(name="id", dtype="INTEGER", nullable=False)
    return {"mydb": {table_name: t}}


@pytest.fixture
def checker():
    return PlausibilityChecker(zero_row_threshold=1_000, numeric_tolerance=3.0)


def test_plausible_result(checker):
    result = _result([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
    pr = checker.check(result, {}, "SELECT id FROM customers")
    assert pr.is_plausible is True
    assert pr.failure_mode == FailureMode.PLAUSIBLE
    assert pr.score == 1.0


def test_zero_rows_small_table(checker):
    result = _result([])
    schema = _schema("orders", row_count=5)
    pr = checker.check(result, schema, "SELECT id FROM orders")
    assert pr.score < 1.0
    assert pr.failure_mode == FailureMode.EMPTY_RESULT


def test_zero_rows_large_table_warns(checker):
    result = _result([])
    schema = _schema("orders", row_count=10_000)
    pr = checker.check(result, schema, "SELECT id FROM orders")
    assert pr.failure_mode == FailureMode.WRONG_JOIN_KEY
    assert pr.score == pytest.approx(0.2)
    assert len(pr.warnings) > 0


def test_zero_rows_table_not_in_query(checker):
    result = _result([])
    schema = _schema("orders", row_count=50_000)
    # Table name "orders" not in query → no large-table warning
    pr = checker.check(result, schema, "SELECT id FROM customers")
    assert pr.failure_mode == FailureMode.EMPTY_RESULT


def test_numeric_outlier_warns(checker):
    col = ColumnMeta(name="amount", dtype="REAL", nullable=True, p95=100.0)
    table = TableMeta(name="orders", row_count_approx=500)
    table.columns["amount"] = col
    schema = {"mydb": {"orders": table}}

    # 500.0 > 3.0 * 100.0 = 300.0 → outlier
    result = _result([{"amount": 500.0}])
    pr = checker.check(result, schema, "SELECT amount FROM orders")
    assert len(pr.warnings) > 0
    assert pr.failure_mode == FailureMode.SCHEMA_MISMATCH


def test_within_range_no_warning(checker):
    col = ColumnMeta(name="amount", dtype="REAL", nullable=True, p95=100.0)
    table = TableMeta(name="orders", row_count_approx=500)
    table.columns["amount"] = col
    schema = {"mydb": {"orders": table}}

    result = _result([{"amount": 250.0}])  # 250 < 300 → ok
    pr = checker.check(result, schema, "SELECT amount FROM orders")
    assert pr.failure_mode == FailureMode.PLAUSIBLE


def test_score_degrades_with_warnings(checker):
    col = ColumnMeta(name="amount", dtype="REAL", nullable=True, p95=10.0)
    table = TableMeta(name="orders", row_count_approx=500)
    table.columns["amount"] = col
    schema = {"mydb": {"orders": table}}

    rows = [{"amount": 999.0 * i} for i in range(1, 8)]  # many outliers
    result = _result(rows)
    pr = checker.check(result, schema, "SELECT amount FROM orders")
    assert pr.score < 1.0
    assert pr.score >= 0.5  # score floor


def test_is_plausible_border():
    pr = PlausibilityResult(score=0.5, failure_mode=FailureMode.PLAUSIBLE)
    assert pr.is_plausible is True

    pr2 = PlausibilityResult(score=0.49, failure_mode=FailureMode.WRONG_JOIN_KEY)
    assert pr2.is_plausible is False
