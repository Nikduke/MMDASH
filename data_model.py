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
    # Construct composite key for column dimension.  Excel appears to
    # concatenate parts of the case and bus identifiers (e.g., C21_S1_230HF2).
    # Here we simply join ``Case name`` and ``Bus name`` with an underscore.
    df = df.copy()
    df["Case_Bus"] = df["Case name"].astype(str) + "_" + df["Bus name"].astype(str)
    return df


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
    """Return sorted bus names matching any of the provided voltages.

    Parameters
    ----------
    voltage_list : Iterable or None
        One or more voltage levels. If ``None`` or empty, all bus names are
        returned.

    Returns
    -------
    list
        Sorted unique bus names that have a ``Bus voltage [kV]`` present in
        ``voltage_list``.
    """
    df = get_data()
    if not voltage_list:
        bus_names = df["Bus name"].unique()
    else:
        bus_names = df[df["Bus voltage [kV]"].isin(voltage_list)]["Bus name"].unique()
    return sorted(bus_names)
