"""Hidden suite for own/coding_ledger. The agent never sees this file."""
import csv, json, subprocess
from pathlib import Path

PY, LEDGER, STORE = "/opt/tools/bin/python", "/app/ledger.py", Path("/app/ledger.json")


def run(*args):
    return subprocess.run([PY, LEDGER, *args], capture_output=True, text=True, cwd="/app")


def fresh():
    STORE.unlink(missing_ok=True)


def lines(out):
    return out.stdout.strip().splitlines()


def test_add_formats_amount_to_two_decimals():
    fresh()
    assert lines(run("add", "3", "food", "tea", "--date", "2026-03-01")) == ["added #1 3.00 food"]
    assert lines(run("add", "0.5", "food", "gum", "--date", "2026-03-02")) == ["added #2 0.50 food"]


def test_ids_do_not_repeat_after_delete():
    fresh()
    run("add", "1.00", "a", "x", "--date", "2026-03-01")
    run("add", "2.00", "b", "y", "--date", "2026-03-02")
    assert lines(run("delete", "1")) == ["deleted #1"]
    out = run("add", "3.00", "c", "z", "--date", "2026-03-03")
    assert out.stdout.strip() == "added #3 3.00 c", out.stdout


def test_list_is_oldest_first_and_totals():
    fresh()
    run("add", "10.00", "food", "late lunch", "--date", "2026-04-10")
    run("add", "5.00", "food", "early snack", "--date", "2026-04-01")
    assert lines(run("list")) == ["#2 2026-04-01 5.00 food early snack", "#1 2026-04-10 10.00 food late lunch", "total 15.00"]


def test_list_empty_prints_only_total():
    fresh()
    assert lines(run("list")) == ["total 0.00"]


def test_list_filters_by_category_and_month():
    fresh()
    run("add", "10.00", "food", "a", "--date", "2026-05-02")
    run("add", "20.00", "travel", "b", "--date", "2026-05-03")
    run("add", "30.00", "food", "c", "--date", "2026-06-04")
    assert lines(run("list", "--category", "food")) == ["#1 2026-05-02 10.00 food a", "#3 2026-06-04 30.00 food c", "total 40.00"]
    assert lines(run("list", "--month", "2026-05")) == ["#1 2026-05-02 10.00 food a", "#2 2026-05-03 20.00 travel b", "total 30.00"]


def test_refunds_reduce_totals_and_appear():
    fresh()
    run("add", "40.00", "food", "dinner", "--date", "2026-07-01")
    run("add", "-15.50", "food", "refund", "--date", "2026-07-02")
    assert lines(run("list"))[-1] == "total 24.50"
    assert lines(run("balance")) == ["food 24.50", "total 24.50"]


def test_balance_hides_zero_categories_and_sorts():
    fresh()
    run("add", "10.00", "zeta", "a", "--date", "2026-08-01")
    run("add", "-10.00", "zeta", "b", "--date", "2026-08-02")
    run("add", "5.00", "alpha", "c", "--date", "2026-08-03")
    assert lines(run("balance")) == ["alpha 5.00", "total 5.00"]


def test_balance_month_filter():
    fresh()
    run("add", "7.00", "food", "a", "--date", "2026-09-01")
    run("add", "9.00", "food", "b", "--date", "2026-10-01")
    assert lines(run("balance", "--month", "2026-10")) == ["food 9.00", "total 9.00"]


def test_money_is_exact_over_many_rows():
    fresh()
    for _ in range(30):
        run("add", "0.10", "cents", "x", "--date", "2026-11-01")
    assert lines(run("list"))[-1] == "total 3.00"


def test_export_csv_quotes_commas_and_orders():
    fresh()
    run("add", "12.00", "food", "lunch, with tip", "--date", "2026-12-02")
    run("add", "3.00", "travel", "bus", "--date", "2026-12-01")
    out = run("export", "/app/exp.csv")
    assert out.stdout.strip() == "exported 2 rows", out.stdout
    rows = list(csv.reader(Path("/app/exp.csv").read_text().splitlines()))
    assert rows[0] == ["id", "date", "amount", "category", "description"]
    assert rows[1] == ["2", "2026-12-01", "3.00", "travel", "bus"]
    assert rows[2] == ["1", "2026-12-02", "12.00", "food", "lunch, with tip"]


def test_unknown_id_and_bad_input_exit_2():
    fresh()
    out = run("delete", "42")
    assert out.returncode == 2 and "no expense #42" in out.stderr
    assert run("add", "1.234", "food", "x").returncode == 2
    assert run("add", "5", "food", "x", "--date", "2026-13-40").returncode == 2


def test_store_survives_and_stays_valid_json():
    fresh()
    run("add", "8.00", "food", "persisted", "--date", "2026-01-09")
    data = json.loads(STORE.read_text())
    assert data, "ledger.json should hold the expense"
    assert lines(run("list"))[0].startswith("#1 2026-01-09 8.00 food persisted")
