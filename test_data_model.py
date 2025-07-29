"""
Unit tests for data_model.py.

These tests validate that the pivot functions implemented in
``data_model.py`` produce the same results as explicit pandas
groupby/aggregation calls.  They do not compare against the Excel workbook
directly (mapping between Excel pivot column labels and CSV fields is not
bijective), but they verify that the aggregation logic is correct.

Run with:

    pytest -q

"""

import pandas as pd

import data_model as dm


def _manual_pivot(df: pd.DataFrame, index: str, values: str, agg: str) -> pd.DataFrame:
    """Helper to compute a pivot via groupby/unstack for comparison."""
    grouped = df.groupby([index, "Case_Bus"])[values]
    if agg == "max":
        result = grouped.max().unstack()
    elif agg == "min":
        result = grouped.min().unstack()
    else:
        raise ValueError(f"Unsupported agg: {agg}")
    return result


def test_lg_inst_max():
    df = dm.get_data()
    pivot = dm.lg_inst_max()
    expected = _manual_pivot(df, "Tswitch_a [s]", "LGp [pu]", "max")
    # Align indices and columns
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_ll_inst_max():
    df = dm.get_data()
    pivot = dm.ll_inst_max()
    expected = _manual_pivot(df, "Tswitch_a [s]", "LLp [pu]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_lg_seek():
    df = dm.get_data()
    pivot = dm.lg_seek()
    expected = _manual_pivot(df, "Run#", "LGp [pu]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_ll_seek():
    df = dm.get_data()
    pivot = dm.ll_seek()
    expected = _manual_pivot(df, "Run#", "LLp [pu]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_lg_rms_max():
    df = dm.get_data()
    pivot = dm.lg_rms_max()
    expected = _manual_pivot(df, "Tswitch_a [s]", "LGr [pu]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_ll_rms_max():
    df = dm.get_data()
    pivot = dm.ll_rms_max()
    expected = _manual_pivot(df, "Tswitch_a [s]", "LLr [pu]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_lg_rms_min():
    df = dm.get_data()
    pivot = dm.lg_rms_min()
    expected = _manual_pivot(df, "Tswitch_a [s]", "LGrm [pu]", "min")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_ll_rms_min():
    df = dm.get_data()
    pivot = dm.ll_rms_min()
    expected = _manual_pivot(df, "Tswitch_a [s]", "LLrm [pu]", "min")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_initial_conditions():
    df = dm.get_data()
    series = dm.initial_conditions()
    expected = df.groupby("Case_Bus")["LLs [pu]"].max()
    expected = expected.reindex(index=series.index)
    assert series.equals(expected)


def test_tov_dur():
    df = dm.get_data()
    pivot = dm.tov_dur()
    expected = _manual_pivot(df, "Run#", "TOV_dur [s]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_filtering():
    df = dm.get_data()
    # Pick a single case and bus for filtering
    case = df["Case name"].iloc[0]
    bus = df["Bus name"].iloc[0]
    filters = {"Case name": {case}, "Bus name": {bus}}
    pivot = dm.lg_inst_max(filters)
    # manual filter
    df_f = df[(df["Case name"] == case) & (df["Bus name"] == bus)]
    expected = _manual_pivot(df_f, "Tswitch_a [s]", "LGp [pu]", "max")
    expected = expected.reindex(index=pivot.index, columns=pivot.columns)
    assert pivot.equals(expected)


def test_get_filter_options_keys():
    options = dm.get_filter_options()
    expected_keys = {
        "Case name",
        "Bus name",
        "Run#",
        "Fault_type",
        "Bus voltage [kV]",
        "Tswitch_a [s]",
        "Tswitch_b [s]",
        "Tswitch_c [s]",
    }
    assert set(options.keys()) == expected_keys