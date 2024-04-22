import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input
from itertools import chain

def draw_graphs(grouped: pd.DataFrame, cat: str, col: str, prop=False, comp="no", disp=["Q1", "Med", "Q3"]):
    q1 = grouped.quantile(0.25).unstack()
    med = grouped.median().unstack()
    q3 = grouped.quantile(0.75).unstack()
    loss = grouped.count()
    colors = ["red", "blue", "green", "magenta"]


    if col == "Loss":

        fig = go.Figure()

        fig.add_traces([go.Scatter(x=loss.xs(lev, level=1).loc["upper"].index, y=((1727/len(loss.index.levels[1]))-loss.xs(lev, level=1).loc["upper"])/(1727/len(loss.index.levels[1])), legendgroup=lev, name=lev, line=dict(color=colors[i], dash="solid")) for i, lev in enumerate(loss.index.levels[1])])

        fig.update_layout(yaxis_title=col, xaxis_title="Time", autosize=False, width=750, height=750, title_text=f"Loss w.r.t. {cat} {f'compared to {comp}' * (comp != 'no')}")
        # fig.add_trace(go.Scatter(x=loss.index.levels[2], y=1-loss.groupby("time").sum()/(1727*3), name="Loss", line=dict(color="black", dash="solid")))
        # fig.update_layout(yaxis_title="Loss", xaxis_title="Time", autosize=False, width=750, height=750,title_text=f"Loss vs time")

    else:
        fig = make_subplots(rows=1,
                            cols=3,
                            shared_yaxes=True,
                            subplot_titles=("Upper Class", "Middle Class", "Lower Class"),
                            y_title=col,
                            x_title="Time")
        if comp != "no":
            q1 = pd.DataFrame(list(chain(
                *[[(q1.loc[x] - q1.loc[clas, comp]).rename(x) for x in q1.index if clas in x] for clas in
                  ["lower", "middle", "upper"]])))
            med = pd.DataFrame(list(chain(
                *[[(med.loc[x] - med.loc[clas, comp]).rename(x) for x in med.index if clas in x] for clas in
                  ["lower", "middle", "upper"]])))
            q3 = pd.DataFrame(list(chain(
                *[[(q3.loc[x] - q3.loc[clas, comp]).rename(x) for x in q3.index if clas in x] for clas in
                  ["lower", "middle", "upper"]])))
            q1 = q1[~q1.index.duplicated(keep='last')]
            med = med[~med.index.duplicated(keep='last')]
            q3 = q3[~q3.index.duplicated(keep='last')]
        if "Q1" in disp:
            fig.add_traces(
                [go.Scatter(x=q1.columns, y=q1.loc["upper", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["upper"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, name="Q1 " + lev, line=dict(color=colors[i], dash="dash")) for i, lev in enumerate(q1.loc["upper"].index)], rows=1, cols=1).add_traces(
                [go.Scatter(x=q1.columns, y=q1.loc["middle", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["middle"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, showlegend=False, line=dict(color=colors[i], dash="dash")) for i, lev in enumerate(q1.loc["middle"].index)], rows=1, cols=2).add_traces(
                [go.Scatter(x=q1.columns, y=q1.loc["lower", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["lower"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, showlegend=False, line=dict(color=colors[i], dash="dash")) for i, lev in enumerate(q1.loc["lower"].index)], rows=1, cols=3)
        if "Med" in disp:
            fig.add_traces(
                [go.Scatter(x=med.columns, y=med.loc["upper", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["upper"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, name="Median " + lev, line=dict(color=colors[i], dash="solid")) for i, lev in enumerate(med.loc["upper"].index)], rows=1, cols=1).add_traces(
                [go.Scatter(x=med.columns, y=med.loc["middle", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["middle"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, showlegend=False, line=dict(color=colors[i], dash="solid")) for i, lev in enumerate(med.loc["middle"].index)], rows=1, cols=2).add_traces(
                [go.Scatter(x=med.columns, y=med.loc["lower", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["lower"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, showlegend=False, line=dict(color=colors[i], dash="solid")) for i, lev in enumerate(med.loc["lower"].index)], rows=1, cols=3)
        if "Q3" in disp:
            fig.add_traces(
                [go.Scatter(x=q3.columns, y=q3.loc["upper", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["upper"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, name="Q3 " + lev, line=dict(color=colors[i], dash="dot")) for i, lev in enumerate(q3.loc["upper"].index)], rows=1, cols=1).add_traces(
                [go.Scatter(x=q3.columns, y=q3.loc["middle", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["middle"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, showlegend=False, line=dict(color=colors[i], dash="dot")) for i, lev in enumerate(q3.loc["middle"].index)], rows=1, cols=2).add_traces(
                [go.Scatter(x=q3.columns, y=q3.loc["lower", lev], marker_size=5+(15*loss.xs(lev, level=1).loc["lower"]/(1727/len(loss.index.levels[1]))), legendgroup=lev, showlegend=False, line=dict(color=colors[i], dash="dot")) for i, lev in enumerate(q3.loc["lower"].index)], rows=1, cols=3)

        fig.update_layout(autosize=True, title_text=f"{col} w.r.t. {cat} {'prop. to Money in Circulation ' * prop}{f'compared to {comp}' * (comp != 'no')}")

    return fig



def main():
    f_data = pd.read_pickle("filtered_data.pkl")
    data = pd.read_pickle("raw_data.pkl")

    # graph col with respect to cat
    # possible cat:     'alpha2', 'eos1', 'eos2', 'robot_growth',
    #                   'income_inequality', 'rate_k', 'rate_z'
    #
    # possible col:     'Convergence Error', 'Money in Circulation',
    #                   'Firm Money', 'Leftover Goods', 'Firm Labor Demand',
    #                   'Firm Capital Demand', 'Firm Robot Demand', 'Robot Growth', 'Price',
    #                   'Wage', 'Consumer Start Moneys', 'Utils', 'Times Worked', 'Leisure',
    #                   'Money Earned', 'Goods Purchased', 'Money Spent', 'Money Saved',
    #                   'Interest Earned'

    # mpl_draw_graphs(data, f_data, "income_inequality", "Wage", comp=True)

    draw_graphs(f_data, "alpha2", "Money Saved", prop=True)



def dash_main():
    app = Dash(__name__)

    df = pd.read_pickle("filtered_data.pkl")

    df = df[df["time"] < 6]
    df = df.astype(float, errors='ignore')


    app.layout = html.Div([
        html.Div(children=[
            html.Div(children=[
                html.Label('Category'),
                dcc.Dropdown(['alpha2', 'eos1', 'eos2', 'robot_growth', 'income_inequality', 'rate_k', 'rate_z'],
                         value='alpha2', id="category")
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Label('Column'),
                dcc.Dropdown(
                    ['Loss', 'Convergence Error', 'Money in Circulation', 'Firm Money', 'Leftover Goods', 'Firm Labor Demand',
                     'Firm Capital Demand', 'Firm Robot Demand', 'Robot Growth', 'Price', 'Wage', 'Consumer Start Moneys',
                     'Utils', 'Times Worked', 'Leisure', 'Money Earned', 'Goods Purchased', 'Money Spent', 'Money Saved',
                     'Interest Earned'], 'Loss', id="column"),
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Label("Proportionate to Money in Circulation"),
                dcc.Checklist(['Proportionate'], [], id="prop"),
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Label("Comparison"),
                dcc.RadioItems(options=["no"], value="no", id="comp"),
            ], style={'padding': 10, 'flex': 1}),

            html.Div(children=[
                html.Label("Display"),
                dcc.Checklist(options=["Q1", "Med", "Q3"], value=["Q1", "Med", "Q3"], id="disp"),
            ], style={'padding': 10, 'flex': 1})
            ], style={'display': 'flex', 'flexDirection': 'row'}),

        html.Br(),
        dcc.Graph(id="graph", style={"height": "750px"})

    ])

    @callback(
        [Output("graph", "figure"),
        Output("comp", "options")],
        [Input("category", "value"),
        Input("column", "value"),
        Input("prop", "value"),
        Input("comp", "value"),
        Input("disp", "value")]
    )
    def update_output(cat, col, prop, comp, disp):
        if col == "Loss":
            df["obs"] = df["Money in Circulation"]
        elif prop:
            df["obs"] = df[col] / df["Money in Circulation"]
        else:
            df["obs"] = df[col]
        grouped = df.groupby(["class", cat, "time"])["obs"]
        return (draw_graphs(grouped, cat, col, "Proportionate" in prop, comp, disp),
                [{"label": x, "value": x} for x in grouped.apply(lambda x: x).unstack(level=1).columns] + [{'label': 'no', 'value': "no"}])

    app.run_server(debug=True)


if __name__ == "__main__":
    dash_main()