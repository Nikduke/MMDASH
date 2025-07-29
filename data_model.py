"""
data_model.py
================

This module encapsulates all data handling and business logic required by the
interactive dashboard.  It reads the raw CSV once and exposes a set of
functions to produce pandas pivot tables mirroring the Excel workbook
(`Overvoltages & Initial voltages.xlsx`).  Each function corresponds to a
worksheet/pivot in the original workbook and contains a docstring stating
which sheet it emulates.  Unit tests (see ``tests/test_data_model.py``)
compare the results of these functions against either hand‑computed
aggregations or the original Excel values.

Key design points:

* ``get_data()`` caches the loaded ``pandas.DataFrame`` so that repeated
  calls do not re‑read the CSV from disk.  If the CSV grows (e.g., new
  rows or columns are appended), the cache can be invalidated by
  restarting the application or adding a manual reload mechanism.
* A convenience ``Case_Bus`` column is created by concatenating
  ``Case name`` and ``Bus name``.  In the source Excel book the column
  labels appear to be a composite of case and bus information.  This
  implementation chooses a simple, repeatable pattern; if a different
  composition is required, adjust ``_build_case_bus`` accordingly.
* Filtering is handled via the ``filters`` parameter.  Each pivot
  function accepts a dictionary mapping column names to a set of allowed
  values.  Only rows matching all specified criteria are retained.
* All aggregation functions rely on pandas' ``pivot_table`` with
  ``aggfunc`` set to ``max`` or ``min`` as appropriate.  Missing
  combinations will produce ``NaN`` values which Dash can handle.

Example usage::

    from data_model import lg_inst_max, get_filter_options
    df = lg_inst_max({"Case name": {"C21_S1_161BB2"}, "Run#": {1, 2}})
    options = get_filter_options()

"""

from __future__ import annotations

import functools
from typing import Dict, Iterable, Optional, Set

import pandas as pd

# Columns used throughout the dashboard
_METRIC_COLS = [
    "LGp [pu]",
    "LLp [pu]",
    "LGr [pu]",
    "LLr [pu]",
    "TOV_dur [s]",
    "LLs [pu]",
]

CSV_PATH = "MM results.csv"


@functools.lru_cache(maxsize=1)
def get_data() -> pd.DataFrame:
    """Load and cache the raw CSV data as a DataFrame.

    The first invocation reads ``CSV_PATH`` from disk.  Subsequent calls
    return the cached DataFrame.  The caller should not mutate the
    returned object.

    Returns
    -------
    pandas.DataFrame
        The loaded data with an additional ``Case_Bus`` column.
    """
    df = pd.read_csv(CSV_PATH)
    df = df.copy()
    # Round switching time once to avoid repeated float comparisons
    if "Tswitch_a [s]" in df.columns:
        df["Tswitch_a [s]"] = df["Tswitch_a [s]"].round(3)

    # Construct composite key for column dimension
    df["Case_Bus"] = df["Case name"].astype(str) + "_" + df["Bus name"].astype(str)

    # Pre-compute undervoltage columns used in several graphs
    if "LGr [pu]" in df.columns:
        df["LG_UV"] = 1 - df["LGr [pu]"]
    if "LLr [pu]" in df.columns:
        df["LL_UV"] = 1 - df["LLr [pu]"]

    return df


def refresh_data() -> None:
    """Clear cached data and filter options."""
    get_data.cache_clear()
    get_filter_options.cache_clear()
    _cached_bus_names.cache_clear()
    get_case_token_info.cache_clear()


@functools.lru_cache(maxsize=1)
def get_case_token_info() -> tuple[list[list[str]], dict[str, list[str]]]:
    """Return token options and mapping for case names.

    The CSV's ``Case name`` column contains structured tokens separated by
    underscores. This function parses all unique case names and groups the
    tokens by their position.

    Returns
    -------
    tuple
        ``(token_options, token_map)`` where ``token_options`` is a list of
        unique values for each token position and ``token_map`` maps each case
        name to its list of tokens (truncated to the shortest length).
    """
    df = get_data()
    cases = sorted(df["Case name"].dropna().unique())
    split_cases = [c.split("_") for c in cases]
    if not split_cases:
        return [], {}
    min_len = min(len(parts) for parts in split_cases)
    token_options: list[list[str]] = []
    for idx in range(min_len):
        token_options.append(sorted({parts[idx] for parts in split_cases}))
    token_map = {case: parts[:min_len] for case, parts in zip(cases, split_cases)}
    return token_options, token_map


def filter_cases_by_parts(selected_parts: dict[int, set[str]]) -> list[str]:
    """Return case names matching all selected token parts."""
    options, token_map = get_case_token_info()
    if not token_map:
        return []
    result = []
    for case, tokens in token_map.items():
        match = True
        for idx, allowed in selected_parts.items():
            if allowed and tokens[idx] not in allowed:
                match = False
                break
        if match:
            result.append(case)
    return sorted(result)


def _apply_filters(
    df: pd.DataFrame, filters: Optional[Dict[str, Iterable]] = None
) -> pd.DataFrame:
    """Return a subset of ``df`` according to provided filters.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    filters : dict or None
        A dictionary where keys are column names of ``df`` and values are
        iterables of accepted values.  Rows where the column value is not
        contained in the iterable are dropped.  If ``filters`` is None or
        empty, the original DataFrame is returned.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame.
    """
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, allowed in filters.items():
        if allowed is None:
            continue
        allowed_set: Set = set(allowed)
        mask &= df[col].isin(allowed_set)
    return df[mask]


def lg_inst_max(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum single‑phase instantaneous overvoltages.

    Replicates sheet **LG_Inst_max** in the original workbook.  The
    underlying metric corresponds to ``LGp [pu]`` in the CSV.  Rows are
    indexed by ``Tswitch_a [s]`` and columns are ``Case_Bus``.  The
    aggregation function is ``max``.

    Parameters
    ----------
    filters : dict, optional
        Filters applied before pivoting (see :func:`_apply_filters`).

    Returns
    -------
    pandas.DataFrame
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Tswitch_a [s]",
        columns="Case_Bus",
        values="LGp [pu]",
        aggfunc="max",
    )
    return pivot.sort_index()


def ll_inst_max(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum line‑to‑line instantaneous overvoltages.

    Replicates sheet **LL_inst_max**.  Uses ``LLp [pu]`` as the value.

    Parameters
    ----------
    filters : dict, optional
        See :func:`lg_inst_max`.

    Returns
    -------
    pandas.DataFrame
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Tswitch_a [s]",
        columns="Case_Bus",
        values="LLp [pu]",
        aggfunc="max",
    )
    return pivot.sort_index()


def lg_seek(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum single‑phase instantaneous overvoltages by run.

    This corresponds to sheet **LG_seek**, where rows index runs (``Run#``)
    rather than switching times.

    Parameters
    ----------
    filters : dict, optional
        See :func:`lg_inst_max`.

    Returns
    -------
    pandas.DataFrame
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Run#",
        columns="Case_Bus",
        values="LGp [pu]",
        aggfunc="max",
    )
    return pivot.sort_index()


def ll_seek(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum line‑to‑line instantaneous overvoltages by run.

    Corresponds to sheet **LL_seek**.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Run#",
        columns="Case_Bus",
        values="LLp [pu]",
        aggfunc="max",
    )
    return pivot.sort_index()


def lg_rms_max(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum single‑phase RMS overvoltages.

    Replicates sheet **LG_RMS_max**.  Uses ``LGr [pu]`` as value and
    ``max`` as aggregator.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Tswitch_a [s]",
        columns="Case_Bus",
        values="LGr [pu]",
        aggfunc="max",
    )
    return pivot.sort_index()


def ll_rms_max(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum line‑to‑line RMS overvoltages.

    Corresponds to sheet **LL_RMS_max**.  Uses ``LLr [pu]`` and ``max``.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Tswitch_a [s]",
        columns="Case_Bus",
        values="LLr [pu]",
        aggfunc="max",
    )
    return pivot.sort_index()


def lg_rms_min(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of minimum single‑phase RMS undervoltages.

    Replicates sheet **LG_RMS_min**.  Uses ``LGrm [pu]`` and ``min``.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Tswitch_a [s]",
        columns="Case_Bus",
        values="LGrm [pu]",
        aggfunc="min",
    )
    return pivot.sort_index()


def ll_rms_min(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of minimum line‑to‑line RMS undervoltages.

    Corresponds to sheet **LL_RMS_min**.  Uses ``LLrm [pu]`` and ``min``.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Tswitch_a [s]",
        columns="Case_Bus",
        values="LLrm [pu]",
        aggfunc="min",
    )
    return pivot.sort_index()


def initial_conditions(filters: Optional[Dict[str, Iterable]] = None) -> pd.Series:
    """Return a Series of maximum initial voltages by case/bus.

    Replicates sheet **Initial conditions**.  Excel groups by the
    ``Case_Bus`` composite and takes the maximum of ``LLs [pu]``.

    Parameters
    ----------
    filters : dict, optional
        See :func:`lg_inst_max`.

    Returns
    -------
    pandas.Series
        Index is ``Case_Bus``, values are maximum ``LLs [pu]``.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    ser = df_filtered.groupby("Case_Bus")["LLs [pu]"].max()
    return ser.sort_index()


def tov_dur(filters: Optional[Dict[str, Iterable]] = None) -> pd.DataFrame:
    """Return a pivot of maximum TOV duration by run and case/bus.

    The original sheet **TOV_dur** indexes rows by a set of integers
    (31–39).  Because the raw CSV does not contain such a field, this
    function groups by ``Run#`` instead.  If the dataset later includes
    another column determining TOV duration categories, adjust the
    ``index`` argument accordingly.

    Parameters
    ----------
    filters : dict, optional
        See :func:`lg_inst_max`.

    Returns
    -------
    pandas.DataFrame
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index="Run#",
        columns="Case_Bus",
        values="TOV_dur [s]",
        aggfunc="max",
    )
    return pivot.sort_index()


@functools.lru_cache(maxsize=1)
def get_filter_options() -> Dict[str, list]:
    """Compute unique values for each supported filter field.

    Returns a mapping from column name to a sorted list of unique values.
    This is useful for populating dropdowns in the Dash layout.

    Returns
    -------
    dict
        Keys: 'Case name', 'Bus name', 'Run#', 'Fault_type', 'Bus voltage [kV]',
        'Tswitch_a [s]', 'Tswitch_b [s]', 'Tswitch_c [s]'.  Values: lists of
        unique entries, sorted where appropriate.
    """
    df = get_data()
    fields = [
        "Case name",
        "Bus name",
        "Run#",
        "Fault_type",
        "Bus voltage [kV]",
        "Tswitch_a [s]",
        "Tswitch_b [s]",
        "Tswitch_c [s]",
    ]
    options: Dict[str, list] = {}
    for field in fields:
        values = df[field].dropna().unique().tolist()
        # sort numerically if possible
        try:
            options[field] = sorted(values)
        except Exception:
            options[field] = sorted(values, key=str)
    return options


def get_bus_names_for_voltage(voltage_list: Optional[Iterable]) -> list:
    """Return sorted bus names matching any of the provided voltages."""

    if not voltage_list:
        key = None
    else:
        key = tuple(sorted(voltage_list))
    return _cached_bus_names(key)


@functools.lru_cache(maxsize=32)
def _cached_bus_names(voltage_tuple: Optional[tuple]) -> list:
    df = get_data()
    if voltage_tuple is None:
        bus_names = df["Bus name"].unique()
    else:
        bus_names = df[df["Bus voltage [kV]"].isin(voltage_tuple)]["Bus name"].unique()
    return sorted(bus_names)


def get_initial_voltage(
    x_axis: str,
    filters: Optional[Dict[str, Iterable]] = None,
) -> pd.DataFrame:
    """Return a pivot of initial voltages at fault inception.

    Parameters
    ----------
    x_axis : {"Run#", "Tswitch_a [s]"}
        Field to use for the row index of the pivot table.
    filters : dict, optional
        Optional filters applied before pivoting. Only the columns ``Case name``,
        ``Bus voltage [kV]`` and ``Bus name`` are typically used.

    Returns
    -------
    pandas.DataFrame
        Pivoted table with ``Bus name`` columns and initial voltage values.
    """
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    value_col = "V0 [pu]" if "V0 [pu]" in df_filtered.columns else "LLs [pu]"
    pivot = df_filtered.pivot_table(
        index=x_axis,
        columns="Bus name",
        values=value_col,
        aggfunc="max",
    )
    return pivot.sort_index()


def get_lg_undervoltage(
    x_axis: str,
    filters: Optional[Dict[str, Iterable]] = None,
) -> pd.DataFrame:
    """Return a pivot of single-phase RMS undervoltages (1 - LGr)."""
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index=x_axis,
        columns="Bus name",
        values="LG_UV",
        aggfunc="max",
    )
    return pivot.sort_index()


def get_ll_undervoltage(
    x_axis: str,
    filters: Optional[Dict[str, Iterable]] = None,
) -> pd.DataFrame:
    """Return a pivot of line-to-line RMS undervoltages (1 - LLr)."""
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    pivot = df_filtered.pivot_table(
        index=x_axis,
        columns="Bus name",
        values="LL_UV",
        aggfunc="max",
    )
    return pivot.sort_index()


def initial_voltage_by_case(filters: Optional[Dict[str, Iterable]] = None) -> pd.Series:
    """Return maximum initial voltage indexed by ``Case_Bus``."""
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    value_col = "V0 [pu]" if "V0 [pu]" in df_filtered.columns else "LLs [pu]"
    ser = df_filtered.groupby("Case_Bus")[value_col].max()
    return ser.sort_index()


def lg_uv_by_case(filters: Optional[Dict[str, Iterable]] = None) -> pd.Series:
    """Return maximum single-phase RMS undervoltage indexed by ``Case_Bus``."""
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    ser = df_filtered.groupby("Case_Bus")["LG_UV"].max()
    return ser.sort_index()


def ll_uv_by_case(filters: Optional[Dict[str, Iterable]] = None) -> pd.Series:
    """Return maximum line-to-line RMS undervoltage indexed by ``Case_Bus``."""
    df = get_data()
    df_filtered = _apply_filters(df, filters)
    ser = df_filtered.groupby("Case_Bus")["LL_UV"].max()
    return ser.sort_index()
