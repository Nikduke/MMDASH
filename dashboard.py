"""
dashboard.py
============

This module defines a Dash web application that reproduces the
functionality of the Excel dashboard found in
``Overvoltages & Initial voltages.xlsx``.  It uses the helper
functions in ``data_model.py`` to compute pivot tables on the fly and
updates plots and key performance indicators (KPIs) in response to
filter selections.

To run the app locally, execute::

    python dashboard.py

Then open http://127.0.0.1:8050 in a web browser.

"""

from __future__ import annotations

import dash
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

import data_model as dm
import pandas as pd


def create_kpi_card(title: str, value: float, unit: str = "") -> html.Div:
    """Helper to generate a simple KPI card with a title and value."""
    return html.Div(
        [
            html.Div(title, className="kpi-title"),
            html.Div(f"{value:.3f}{unit}", className="kpi-value"),
        ],
        className="kpi-card",
    )


def build_layout() -> html.Div:
    """Construct the Dash layout."""
    # Fetch filter options
    options = dm.get_filter_options()

    # Create dropdowns for each filter
    controls = []
    # Multi-select for Case name
    controls.append(
        html.Div(
            [
                html.Label("Case name"),
                dcc.Dropdown(
                    options=[{"label": v, "value": v} for v in options["Case name"]],
                    value=options["Case name"],
                    multi=True,
                    id="case-filter",
                ),
            ]
        )
    )
    # Multi-select for Bus name
    controls.append(
        html.Div(
            [
                html.Label("Bus name"),
                dcc.Dropdown(
                    options=[{"label": v, "value": v} for v in options["Bus name"]],
                    value=options["Bus name"],
                    multi=True,
                    id="bus-filter",
                ),
            ]
        )
    )
    # Multi-select for Fault type
    controls.append(
        html.Div(
            [
                html.Label("Fault type"),
                dcc.Dropdown(
                    options=[{"label": v, "value": v} for v in options["Fault_type"]],
                    value=options["Fault_type"],
                    multi=True,
                    id="fault-filter",
                ),
            ]
        )
    )
    # Multi-select for Run#
    controls.append(
        html.Div(
            [
                html.Label("Run #"),
                dcc.Dropdown(
                    options=[{"label": str(v), "value": v} for v in options["Run#"]],
                    value=options["Run#"],
                    multi=True,
                    id="run-filter",
                ),
            ]
        )
    )
    # Multi-select for Bus voltage
    controls.append(
        html.Div(
            [
                html.Label("Bus voltage [kV]"),
                dcc.Dropdown(
                    options=[{"label": str(v), "value": v} for v in options["Bus voltage [kV]"]],
                    value=options["Bus voltage [kV]"],
                    multi=True,
                    id="voltage-filter",
                ),
            ]
        )
    )
    # Range slider for Tswitch_a [s]
    tswitch_a_values = options["Tswitch_a [s]"]
    if tswitch_a_values:
        min_ts, max_ts = min(tswitch_a_values), max(tswitch_a_values)
    else:
        min_ts = max_ts = 0
    controls.append(
        html.Div(
            [
                html.Label("Tswitch_a [s] range"),
                dcc.RangeSlider(
                    id="tswitch-filter",
                    min=min_ts,
                    max=max_ts,
                    value=[min_ts, max_ts],
                    step=(max_ts - min_ts) / max(1, len(tswitch_a_values) - 1),
                    tooltip={"always_visible": False, "placement": "top"},
                ),
            ]
        )
    )

    # KPI container (empty, values filled via callback)
    kpi_container = html.Div(id="kpi-container", className="kpi-container")

    # Graph placeholders
    graph_ids = [
        "lg-inst-graph",
        "ll-inst-graph",
        "lg-rms-over-graph",
        "ll-rms-over-graph",
        "lg-rms-under-graph",
        "ll-rms-under-graph",
        "initial-voltage-graph",
        "tov-dur-graph",
    ]
    graph_titles = [
        "LG instantaneous overvoltages vs switching time",
        "LL instantaneous overvoltages vs switching time",
        "LG RMS overvoltages vs switching time",
        "LL RMS overvoltages vs switching time",
        "LG RMS undervoltages vs switching time",
        "LL RMS undervoltages vs switching time",
        "Initial voltages",
        "TOV duration by run",
    ]
    graphs = []
    for g_id, title in zip(graph_ids, graph_titles):
        graphs.append(
            html.Div(
                [
                    html.H4(title),
                    dcc.Graph(id=g_id),
                ],
                className="graph-wrapper",
            )
        )

    # Assemble layout
    layout = html.Div(
        [
            html.H1("Overvoltage Dashboard"),
            html.Div(controls, className="controls"),
            kpi_container,
            html.Div(graphs, className="graphs"),
        ],
        className="container",
    )
    return layout


def main() -> None:
    """Entry point for running the Dash application."""
    app = Dash(__name__)
    app.title = "Overvoltage Dashboard"
    app.layout = build_layout()

    # Define callback for updating graphs and KPIs
    @app.callback(
        [
            Output("kpi-container", "children"),
            Output("lg-inst-graph", "figure"),
            Output("ll-inst-graph", "figure"),
            Output("lg-rms-over-graph", "figure"),
            Output("ll-rms-over-graph", "figure"),
            Output("lg-rms-under-graph", "figure"),
            Output("ll-rms-under-graph", "figure"),
            Output("initial-voltage-graph", "figure"),
            Output("tov-dur-graph", "figure"),
        ],
        [
            Input("case-filter", "value"),
            Input("bus-filter", "value"),
            Input("fault-filter", "value"),
            Input("run-filter", "value"),
            Input("voltage-filter", "value"),
            Input("tswitch-filter", "value"),
        ],
    )
    def update_all(
        case_vals, bus_vals, fault_vals, run_vals, voltage_vals, tswitch_range
    ):
        # Build filters dict for data_model
        filters = {
            "Case name": set(case_vals) if case_vals else None,
            "Bus name": set(bus_vals) if bus_vals else None,
            "Fault_type": set(fault_vals) if fault_vals else None,
            "Run#": set(run_vals) if run_vals else None,
            "Bus voltage [kV]": set(voltage_vals) if voltage_vals else None,
        }
        # Filter by Tswitch_a range separately
        df = dm.get_data()
        df_range = df[df["Tswitch_a [s]"].between(tswitch_range[0], tswitch_range[1])]
        df_range = dm._apply_filters(df_range, filters)

        # Compute KPIs
        def safe_max(series: pd.Series) -> float:
            return float(series.max()) if not series.empty else float("nan")
        def safe_min(series: pd.Series) -> float:
            return float(series.min()) if not series.empty else float("nan")

        kpi_cards = []
        kpi_cards.append(create_kpi_card("Max LG instantaneous [pu]", safe_max(df_range["LGp [pu]"])) )
        kpi_cards.append(create_kpi_card("Max LL instantaneous [pu]", safe_max(df_range["LLp [pu]"])) )
        kpi_cards.append(create_kpi_card("Max LG RMS [pu]", safe_max(df_range["LGr [pu]"])) )
        kpi_cards.append(create_kpi_card("Min LG RMS [pu]", safe_min(df_range["LGrm [pu]"])) )
        kpi_cards.append(create_kpi_card("Max TOV duration [s]", safe_max(df_range["TOV_dur [s]"])) )

        # Generate figures
        fig_lg_inst = _pivot_to_line(dm.lg_inst_max(filters), "Tswitch_a [s]", "LGp [pu]")
        fig_ll_inst = _pivot_to_line(dm.ll_inst_max(filters), "Tswitch_a [s]", "LLp [pu]")
        fig_lg_rms_over = _pivot_to_line(dm.lg_rms_max(filters), "Tswitch_a [s]", "LGr [pu]")
        fig_ll_rms_over = _pivot_to_line(dm.ll_rms_max(filters), "Tswitch_a [s]", "LLr [pu]")
        fig_lg_rms_under = _pivot_to_line(dm.lg_rms_min(filters), "Tswitch_a [s]", "LGrm [pu]")
        fig_ll_rms_under = _pivot_to_line(dm.ll_rms_min(filters), "Tswitch_a [s]", "LLrm [pu]")
        fig_initial = _series_to_bar(dm.initial_conditions(filters))
        fig_tov = _pivot_to_line(dm.tov_dur(filters), "Run#", "TOV_dur [s]")

        return (
            kpi_cards,
            fig_lg_inst,
            fig_ll_inst,
            fig_lg_rms_over,
            fig_ll_rms_over,
            fig_lg_rms_under,
            fig_ll_rms_under,
            fig_initial,
            fig_tov,
        )

    # Run the server
    # Dash versions >=2.7 rename run_server() to run().
    # For backwards compatibility, call run() if available.
    try:
        app.run(debug=False)
    except AttributeError:
        # Fallback for older Dash versions
        app.run_server(debug=False)


def _pivot_to_line(pivot: "pd.DataFrame", x_name: str, y_name: str) -> go.Figure:
    """Convert a wide pivot DataFrame into a line chart figure."""
    fig = go.Figure()
    for col in pivot.columns:
        series = pivot[col].dropna()
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="lines", name=col)
        )
    fig.update_layout(
        xaxis_title=x_name,
        yaxis_title=y_name,
        legend_title="Case_Bus",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _series_to_bar(series: "pd.Series") -> go.Figure:
    """Convert a Series into a bar chart figure."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=series.index,
            y=series.values,
        )
    )
    fig.update_layout(
        xaxis_title="Case_Bus",
        yaxis_title="LLs [pu] (max)",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


if __name__ == "__main__":
    main()
