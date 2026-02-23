import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from crash_course.module_2.atlas.cluster import get_min_max_distance
from crash_course.module_2.atlas.vector_space import VectorSpace
from dash import Dash, Input, Output, dcc, html
from hdbscan.plots import CondensedTree, SingleLinkageTree

logger = logging.getLogger(__name__)


def visualize(
    space: VectorSpace,
    debug=False,
) -> None:
    # todo: this function shouldnt access internal details of vectorspace
    """
    Generate a plot of vectorspace
    """
    WIDTH = 1200
    HEIGHT = 800
    DOT_SIZE = 4
    OPACITY = 0.5
    if debug:
        show_debug_info(space.condensed_tree, space.single_linkage_tree)

    df = space.as_df()
    # graph clusters
    if not space.has_position_3d:
        raise ValueError("3d position has not been set on the VectorSpace")

    fig = px.scatter_3d(
        df,
        x="3d_x",
        y="3d_y",
        z="3d_z",
        opacity=OPACITY,
        color="topic",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data={
            "3d_x": False,
            "3d_y": False,
            "3d_z": False,
        },
    )
    fig.update_traces(marker=dict(size=DOT_SIZE))

    fig.update_layout(
        title=space.title,
        width=WIDTH,
        height=HEIGHT,
        scene=dict(
            bgcolor="black",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )
    app = Dash(__name__)
    app.layout = html.Div([dcc.Graph(figure=fig)])
    app.run(debug=False)


""" ---------------------------------------------------------- """


def show_debug_info(ct: CondensedTree, slt: SingleLinkageTree):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ct.plot(axis=ax1)
    ax1.set_title("Condensed Tree")
    slt.plot(axis=ax2)
    ax2.set_title("Single Linkage Tree")
    plt.tight_layout()
    plt.show()


def build_figure(space: VectorSpace, cut_distance: float) -> go.Figure:
    WIDTH = 1200
    HEIGHT = 800
    DOT_SIZE = 4
    OPACITY = 0.5
    labels = space.single_linkage_tree.get_clusters(
        cut_distance=cut_distance, min_cluster_size=5
    )
    space.df["cluster"] = labels

    palette = px.colors.qualitative.Bold
    traces = []
    for cluster, group in space.df.groupby("cluster"):
        assert isinstance(cluster, (int, np.integer))
        cluster = int(cluster)
        traces.append(
            go.Scatter3d(
                x=group["3d_x"],
                y=group["3d_y"],
                z=group["3d_z"],
                mode="markers",
                name=f"Cluster {cluster}",
                marker=dict(
                    size=DOT_SIZE,
                    color=palette[cluster % len(palette)],
                    opacity=OPACITY,
                ),
                hovertemplate="%{fullData.name}",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=space.title,
        width=WIDTH,
        height=HEIGHT,
        scene=dict(
            bgcolor="black",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )
    return fig


def visualize_dynamic(
    space: VectorSpace,
    debug=False,
) -> None:
    """
    this adds a slider that modifies the cut_distance variable of SingleLinkageTree
    """
    app = Dash(__name__)
    if debug:
        show_debug_info(space.condensed_tree, space.single_linkage_tree)
    if not space.has_position_3d:
        raise ValueError("3d position has not been set on the VectorSpace")

    min_dist, max_dist = get_min_max_distance(space.single_linkage_tree)
    app.layout = html.Div(
        [
            dcc.Slider(
                id="cut-distance-slider",
                min=min_dist,
                max=max_dist,
                step=(max_dist - min_dist) / 1000,
                value=max_dist,
                marks={f: f"{f:.2f}" for f in np.linspace(min_dist, max_dist, 1000)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            dcc.Graph(id="scatter-3d"),
        ]
    )

    @app.callback(
        Output("scatter-3d", "figure"),
        Input("cut-distance-slider", "value"),
    )
    def update(cut_distance):
        return build_figure(space, cut_distance)

    app.run(debug=False)
