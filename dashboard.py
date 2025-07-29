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

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

import data_model as dm
import numpy as np


def create_kpi_table(rows: list[dict]) -> html.Table:
    """Return an HTML table summarizing KPI metrics."""
    header = html.Thead(
        html.Tr(
            [
                html.Th("Metric"),
                html.Th("Value"),
                html.Th("Case name"),
                html.Th("Bus name"),
                html.Th("Run#"),
            ]
        )
    )
    body_rows = []
    for row in rows:
        body_rows.append(
            html.Tr(
                [
                    html.Td(row["Metric"]),
                    html.Td(
                        f"{row['Value']:.3f}" if row["Value"] == row["Value"] else "n/a"
                    ),
                    html.Td(row.get("Case name", "")),
                    html.Td(row.get("Bus name", "")),
                    html.Td(str(row.get("Run#", ""))),
                ]
            )
        )
    return html.Table([header, html.Tbody(body_rows)], className="kpi-table")


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
    # Multi-select for Bus voltage
    controls.append(
        html.Div(
            [
                html.Label("Bus voltage [kV]"),
                dcc.Dropdown(
                    options=[
                        {"label": str(v), "value": v}
                        for v in options["Bus voltage [kV]"]
                    ],
                    value=options["Bus voltage [kV]"],
                    multi=True,
                    id="voltage-filter",
                ),
            ]
        )
    )
    # Multi-select for Bus name (populated based on voltage)
    initial_bus = dm.get_bus_names_for_voltage(options["Bus voltage [kV]"])
    controls.append(
        html.Div(
            [
                html.Label("Bus name"),
                dcc.Dropdown(
                    options=[{"label": b, "value": b} for b in initial_bus],
                    value=initial_bus,
                    multi=True,
                    id="bus-filter",
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
                    min=round(min_ts, 3),
                    max=round(max_ts, 3),
                    value=[round(min_ts, 3), round(max_ts, 3)],
                    step=None,
                    marks={round(v, 3): f"{v:.3f}" for v in tswitch_a_values},
                    tooltip={"always_visible": False, "placement": "bottom"},
                ),
            ]
        )
    )

    # X-axis toggle
    controls.append(
        html.Div(
            [
                html.Label("X-axis"),
                dcc.RadioItems(
                    options=[
                        {"label": "Run#", "value": "Run#"},
                        {"label": "Tswitch_a [s]", "value": "Tswitch_a [s]"},
                    ],
                    value="Run#",
                    id="xaxis-toggle",
                    inline=True,
                ),
            ]
        )
    )
    controls.append(html.Div(html.Button("Refresh", id="refresh-button")))

    # KPI container (empty, values filled via callback)
    kpi_container = html.Div(id="kpi-container", className="kpi-container")

    # Graph placeholders
    graph_ids = [
        "lg-inst-graph",
        "ll-inst-graph",
        "lg-rms-graph",
        "ll-rms-graph",
        "tov-dur-graph",
        "v0-graph",
        "lg-uv-graph",
        "ll-uv-graph",
    ]
    graph_titles = [
        "LG instantaneous overvoltages",
        "LL instantaneous overvoltages",
        "LG RMS overvoltages",
        "LL RMS overvoltages",
        "TOV duration",
        "Initial Voltage [pu] vs Case_Bus",
        "LG RMS Undervoltage [pu]",
        "LL RMS Undervoltage [pu]",
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
    app = Dash(
        __name__,
        external_stylesheets=[
            "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        ],
    )
    app.title = "Overvoltage Dashboard"
    app.layout = build_layout()

    @app.callback(
        [Output("bus-filter", "options"), Output("bus-filter", "value")],
        [Input("voltage-filter", "value"), Input("refresh-button", "n_clicks")],
    )
    def update_bus_options(selected_voltages, n_clicks):
        dm.refresh_data()
        bus_names = dm.get_bus_names_for_voltage(selected_voltages)
        opts = [{"label": b, "value": b} for b in bus_names]
        return opts, bus_names

    # Define callback for updating graphs and KPIs
    @app.callback(
        [
            Output("kpi-container", "children"),
            Output("lg-inst-graph", "figure"),
            Output("ll-inst-graph", "figure"),
            Output("lg-rms-graph", "figure"),
            Output("ll-rms-graph", "figure"),
            Output("tov-dur-graph", "figure"),
            Output("v0-graph", "figure"),
            Output("lg-uv-graph", "figure"),
            Output("ll-uv-graph", "figure"),
        ],
        [
            Input("case-filter", "value"),
            Input("fault-filter", "value"),
            Input("voltage-filter", "value"),
            Input("bus-filter", "value"),
            Input("tswitch-filter", "value"),
            Input("xaxis-toggle", "value"),
            Input("refresh-button", "n_clicks"),
        ],
    )
    def update_all(
        case_vals,
        fault_vals,
        voltage_vals,
        bus_vals,
        tswitch_range,
        xaxis_choice,
        n_clicks,
    ):
        dm.refresh_data()
        # Build filters dict for data_model
        filters = {
            "Case name": set(case_vals) if case_vals else None,
            "Bus name": set(bus_vals) if bus_vals else None,
            "Fault_type": set(fault_vals) if fault_vals else None,
            "Bus voltage [kV]": set(voltage_vals) if voltage_vals else None,
        }
        df = dm.get_data()
        df_range = df[df["Tswitch_a [s]"].between(tswitch_range[0], tswitch_range[1])]
        df_range = dm._apply_filters(df_range, filters)
        if df_range.empty:
            return (
                create_kpi_table([]),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
            )

        metrics = {
            "Max LG instantaneous": "LGp [pu]",
            "Max LL instantaneous": "LLp [pu]",
            "Max LG RMS": "LGr [pu]",
            "Max LL RMS": "LLr [pu]",
            "Max TOV duration": "TOV_dur [s]",
        }
        kpi_rows = []
        for name, col in metrics.items():
            idx = df_range[col].idxmax()
            row = df_range.loc[idx]
            kpi_rows.append(
                {
                    "Metric": name,
                    "Value": row[col],
                    "Case name": row["Case name"],
                    "Bus name": row["Bus name"],
                    "Run#": row["Run#"],
                }
            )

        kpi_table = create_kpi_table(kpi_rows)

        # Aggregate once for all metrics to minimize repeated grouping
        agg_cols = {
            "LGp [pu]": "max",
            "LLp [pu]": "max",
            "LGr [pu]": "max",
            "LLr [pu]": "max",
            "TOV_dur [s]": "max",
            "LLs [pu]": "max",
            "LG_UV": "max",
            "LL_UV": "max",
            "Run#": "first",
            "Case name": "first",
            "Bus name": "first",
        }
        x_col = "Run#" if xaxis_choice == "Run#" else "Tswitch_a [s]"
        grouped = (
            df_range.groupby(["Case_Bus", x_col])
            .agg(agg_cols)
            .reset_index()
            .rename(columns={x_col: "x"})
        )

        def build_fig(col: str, y_label: str) -> go.Figure:
            fig = go.Figure()
            for case_bus, sub in grouped.groupby("Case_Bus"):
                case = sub["Case name"].iloc[0]
                bus = sub["Bus name"].iloc[0]
                customdata = np.stack(
                    [
                        np.full(len(sub), case),
                        np.full(len(sub), bus),
                        sub["Run#"],
                    ],
                    axis=-1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=sub["x"],
                        y=sub[col],
                        mode="lines+markers",
                        name=f"{case}-{bus}",
                        customdata=customdata,
                        hovertemplate="Case name: %{customdata[0]}<br>"
                        "Bus name: %{customdata[1]}<br>"
                        "Run#: %{customdata[2]}<br>"
                        + y_label
                        + ": %{y:.3f}<extra></extra>",
                    )
                )
            fig.update_layout(
                xaxis_title=xaxis_choice,
                yaxis_title=y_label,
                legend_title="Case-Bus",
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            return fig

        def build_case_fig(col: str, y_label: str) -> go.Figure:
            idx = df_range.groupby("Case_Bus")[col].idxmax()
            rows = df_range.loc[idx].sort_values("Case_Bus")
            customdata = np.stack(
                [rows["Case name"], rows["Bus name"], rows["Run#"]],
                axis=-1,
            )
            fig = go.Figure(
                data=go.Scatter(
                    x=rows["Case_Bus"],
                    y=rows[col],
                    mode="markers",
                    customdata=customdata,
                    hovertemplate="Case name: %{customdata[0]}<br>"
                    "Bus name: %{customdata[1]}<br>"
                    "Run#: %{customdata[2]}<br>"
                    + y_label
                    + ": %{y:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title="Case_Bus",
                yaxis_title=y_label,
                height=400,
                margin=dict(l=40, r=20, t=40, b=40),
            )
            return fig

        fig_lg_inst = build_fig("LGp [pu]", "LGp [pu]")
        fig_ll_inst = build_fig("LLp [pu]", "LLp [pu]")
        fig_lg_rms = build_fig("LGr [pu]", "LGr [pu]")
        fig_ll_rms = build_fig("LLr [pu]", "LLr [pu]")
        fig_tov = build_fig("TOV_dur [s]", "TOV_dur [s]")
        fig_v0 = build_case_fig("LLs [pu]", "V0 [pu]")
        fig_lg_uv = build_fig("LG_UV", "1 - LGr [pu]")
        fig_ll_uv = build_fig("LL_UV", "1 - LLr [pu]")

        return (
            kpi_table,
            fig_lg_inst,
            fig_ll_inst,
            fig_lg_rms,
            fig_ll_rms,
            fig_tov,
            fig_v0,
            fig_lg_uv,
            fig_ll_uv,
        )

    # Run the server
    # Dash versions >=2.7 rename run_server() to run().
    # For backwards compatibility, call run() if available.
    try:
        app.run(debug=False)
    except AttributeError:
        # Fallback for older Dash versions
        app.run_server(debug=False)


if __name__ == "__main__":
    main()
