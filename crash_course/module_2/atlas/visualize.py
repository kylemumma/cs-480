import logging

import plotly.express as px

from crash_course.module_2.atlas.vector_space import VectorSpace

logger = logging.getLogger(__name__)


def visualize(
    space: VectorSpace,
    plot_3d=True,
    plot_2d=False,
    width: int = 1200,
    height: int = 800,
    show_noise: bool = True,
    dot_size: int = 4,
    opacity: float = 0.5,
) -> None:
    # todo: this function shouldnt access internal details of vectorspace
    """
    Generate a plot of vectorspace
    """
    if plot_2d:
        logger.warning("2d visualization not supported yet, skipping")
    if plot_3d:
        for col in ["cluster", "3d_x", "3d_y", "3d_z"]:
            if col not in space.df.columns:
                raise ValueError(f"column {col} not found in space.df")
        if "topic" not in space.clusters.columns:
            raise ValueError("column topic not found in space.clusters")
        df = space.df.drop(columns=["topic"], errors="ignore").merge(
            space.clusters[["cluster", "topic"]], on="cluster", how="left"
        )

        # graph clusters
        clusters = df[df["cluster"].astype(str) != "-1"]
        fig = px.scatter_3d(
            clusters,
            x="3d_x",
            y="3d_y",
            z="3d_z",
            opacity=opacity,
            color="topic",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={
                "3d_x": False,
                "3d_y": False,
                "3d_z": False,
            },
        )
        fig.update_traces(marker=dict(size=dot_size))

        if show_noise:
            noise = df[df["cluster"].astype(str) == "-1"]
            fig.add_scatter3d(
                x=noise["3d_x"],
                y=noise["3d_y"],
                z=noise["3d_z"],
                mode="markers",
                marker=dict(
                    color="lightgray", opacity=opacity * 0.7, size=dot_size - 1
                ),
                name="noise",
                hovertemplate="%{fullData.name}",
                hoverlabel=dict(
                    bgcolor="#cccccc",  # Dark background
                    font_color="black",  # Force white text
                ),
            )
        fig.update_layout(
            title=space.title,
            width=width,
            height=height,
            scene=dict(
                bgcolor="black",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
        )

        fig.show()
