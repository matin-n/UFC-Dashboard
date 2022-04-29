import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import ListedColormap
from pathlib import Path

# user defined functions
# create prediction pie chart
def pie_chart_prediction(df) -> px.pie:
    colors = ["rgba(235, 60, 12, 0.60)", "rgba(105, 43, 255, 0.60)"]
    fig = px.pie(
        df,
        values=df.iloc[0].values,
        names=[f"{red_name}", f"{blue_name}"],
        color_discrete_sequence=colors,
        hole=0.75,
    )

    fig.update_traces(
        hovertemplate="%{label}: %{percent}",
        textinfo="percent",
        textfont_size=20,
        showlegend=False,
        title=f"<b>Name Here</b><br>50%",  # TODO: use predicted winner and %
    )
    fig.update_layout(
        font_color="red", # TODO: change color depending on winner
        font_size=25,
    )

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    return fig


# create stats table
def stats_table(red_stats, blue_stats) -> go.Figure:
    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        )
    )
    stats = ["Win - Loss - Draws", "Age", "Height", "Weight", "Reach", "Stance", "Born"]

    header_values = [f"<b>{red_name}</b>", "<b>vs.</b>", f"<b>{blue_name}</b>"]
    cells_values = [red_stats, stats, blue_stats]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_values,
                    line_color="#E1ECF0",
                    fill_color="#e0e0ef",
                    font=dict(
                        color=["#AC302C", "#5B5B5B", "#315D9E"],
                        family="Source Sans Pro",
                        size=36,
                    ),
                    # font_color=["red", "grey", "blue"],
                    align=["right", "center", "left"],
                ),
                cells=dict(
                    values=cells_values,
                    line_color="#e0e0ef",
                    fill_color="#f0f0f5",
                    # f0f0f5
                    align=["right", "center", "left"],
                    height=30,
                    font=dict(
                        color=["#14141A", "#040406", "#14141A"],
                        family="Source Sans Pro",
                        size=18,
                    ),
                    # prefix = [None, "$"],
                ),
            )
        ],
        layout=layout,
    )

    # fig.update_layout(overwrite=False, autosize=True,
    #     font=dict(family="Source Sans Pro", color="#000000")
    # )  # height=400,

    return fig


def two_sided_barchart(red_offensive_stats, blue_offensive_stats) -> go.Figure:
    fig = make_subplots(
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        horizontal_spacing=0,
        shared_yaxes=True,
        rows=1,
        cols=2,
        print_grid=False,
    )

    # x1 = red_offensive_stats.iloc[0].values
    # text1 = [f"{t}%" for t in x1]
    y = ["Submissions", "Takedowns", "Striking"]
    fig.add_trace(
        go.Bar(
            orientation="h",
            x=red_offensive_stats,
            y=y,
            name=red_name,
            # text=text1,
            # customdata=red_offensive_stats.iloc[0].values,
            texttemplate="%{x}%",
            textposition="inside",
            hovertemplate="%{y}: %{x}%",
            marker=dict(
                color="rgba(235, 60, 12, 0.60)",
                line=dict(color="rgba(235, 60, 12, 1.0)", width=1),
            ),
        ),
        row=1,
        col=1,
    )
    # x2 = blue_offensive_stats.iloc[0].values
    fig.add_trace(
        go.Bar(
            orientation="h",
            x=blue_offensive_stats,
            y=y,
            name=blue_name,
            # text=x2,
            texttemplate="%{x}%",
            textposition="inside",
            hovertemplate="%{y}: %{x}%",
            marker=dict(
                color="rgba(105, 43, 255, 0.60)",
                line=dict(color="rgba(105, 43, 255, 1.0)", width=1),
            ),
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # width=550,
        # height=500,
        # autosize=True,
        margin=dict(l=0, r=0, b=0, t=0, pad=4),
        xaxis_showticklabels=True,
        xaxis2_showticklabels=True,
        xaxis_title=red_name,
        xaxis2_title=blue_name,
        yaxis3_showticklabels=False,
    )

    # https://plotly.com/python/subplots/
    max_val = (
        max(
            max(red_offensive_stats),
            max(blue_offensive_stats),
        )
    ) * 1.10
    fig.update_xaxes(range=[max_val, 0], row=1, col=1)
    fig.update_xaxes(range=[0, max_val], row=1, col=2)

    return fig


def prediction_table(styler):

    cmp_white_red = ListedColormap(["#fbfcea", "#d7e7d0", "#AC302C", "#5E120F"])
    cmp_white_blue = ListedColormap(["#fbfcea", "#d7e7d0", "#1B6BAB", "#143D5E"])

    # Hide index
    # styler.hide(axis="index")

    # Sticky Header
    # https://pandas.pydata.org/docs/user_guide/style.html#Sticky-Headers
    styler.set_sticky(axis=1)

    # Format as percentages
    # https://pandas.pydata.org/docs/user_guide/style.html#Formatting-Values
    styler.format(
        "{:.2%}",
        subset=["red_predicted_probability", "blue_predicted_probability"],
        na_rep="",
    )

    # Add bar chart to predicted probability # TODO: What colorscale should I use..?
    # https://pandas.pydata.org/docs/user_guide/style.html#Bar-charts
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    # styler.bar(subset=["red_predicted_probability", "blue_predicted_probability"], align=0, vmin=0, vmax=1, cmap="bone", height=50, width=60, props="width: 120px; border-right: 1px solid black;")
    # styler.bar(subset=["red_predicted_probability", "blue_predicted_probability"], align=0, vmin=0, vmax=1, cmap="coolwarm", height=50, width=60, props="width: 120px; border-right: 1px solid black;")
    styler.bar(
        subset=["red_predicted_probability"],
        align=0,
        vmin=0,
        vmax=1,
        cmap=cmp_white_red,
        height=50,
        width=60,
        # props="width: 120px; border-right: 1px solid black;",
    )
    styler.bar(
        subset=["blue_predicted_probability"],
        align=0,
        vmin=0,
        vmax=1,
        cmap=cmp_white_blue,
        height=50,
        width=60,
        # props="width: 120px; border-right: 1px solid black;",
    )

    # https://pandas.pydata.org/docs/user_guide/style.html#Background-Gradient-and-Text-Gradient

    # Text Gradient on Prediction Column
    # styler.text_gradient(subset=["prediction"],  vmin=0, vmax=1, gmap=df["red_predicted_probability"], cmap="coolwarm")

    # Text Gradient on Red & Blue Predicted Probability
    # styler.text_gradient(subset=["red_predicted_probability", "blue_predicted_probability"],  vmin=0, vmax=1, gmap=df["red_predicted_probability"], cmap="coolwarm")

    # Background Gradient on Prediction Column
    # styler.background_gradient(subset=["prediction"],  vmin=0, vmax=1, gmap=df["red_predicted_probability"], cmap="coolwarm")

    # Background Gradient on Red & Blue Predicted Probability
    # styler.background_gradient(axis=None, vmin=0, vmax=1, cmap="coolwarm")

    # Background Gradient on Red & Blue Predicted Probability (making the colors same w/ gmap)
    # styler.background_gradient(subset=["prediction"],  vmin=0, vmax=1, gmap=df["red_predicted_probability"], cmap="coolwarm")

    # Background Gradient on Red & Blues Predicted Probability (making the colors the same by applying opposite gradient)
    # styler.background_gradient(subset=["red_predicted_probability"],  vmin=0, vmax=1, cmap="coolwarm")
    # styler.background_gradient(subset=["blue_predicted_probability"], vmin=0, vmax=1, cmap="coolwarm_r")

    return styler


# read csv from github repo/local directory
@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(
        Path.joinpath(Path.cwd(), "Predictions", "LogisticRegression.csv"), index_col=0
    )


st.set_page_config(
    page_title="UFC Prediction",
    page_icon="✅",
    # layout="centered",
    layout="wide",
)

df = get_data()

# dashboard title
st.title("UFC Prediction Dashboard")

# top-level filters
event_filter = st.selectbox("Select the Event", pd.unique(df["event"]))
# dataframe filter
df = df[df["event"] == event_filter]

# second-level filter
match_filter = st.selectbox("Select the Match", pd.unique(df["matchup"]))
df_match = df[df["matchup"] == match_filter]

# creating a single-element container
placeholder = st.empty()


# grab name
red_name = df_match["red_name"].item()
blue_name = df_match["blue_name"].item()

# grab predictions
red_pred = df_match["red_predicted_probability"].item()
blue_pred = df_match["blue_predicted_probability"].item()

# stats for table chart
red_stats = (
    df_match[
        [
            "red_wld",
            "red_age",
            "red_weight",
            "red_height",
            "red_reach",
            "red_stance",
            "red_born",
        ]
    ]
    .iloc[0]
    .values
)


blue_stats = (
    df_match[
        [
            "blue_wld",
            "blue_age",
            "blue_weight",
            "blue_height",
            "blue_reach",
            "blue_stance",
            "blue_born",
        ]
    ]
    .iloc[0]
    .values
)

# offensive stats for two sided bar chart
red_offensive_stats = (
    df_match[
        [
            "red_submissions_percentage",
            "red_takedowns_percentage",
            "red_striking_percentage",
        ]
    ]
    .iloc[0]
    .values
)

blue_offensive_stats = (
    df_match[
        [
            "blue_submissions_percentage",
            "blue_takedowns_percentage",
            "blue_striking_percentage",
        ]
    ]
    .iloc[0]
    .values
)


# ploty config
config = {"displayModeBar": False}


with placeholder.container():

    # Event name
    st.header(event_filter)
    # Create two columns for charts
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### Predicted Probability")
        fig = pie_chart_prediction(
            df_match[["red_predicted_probability", "blue_predicted_probability"]]
        )
        st.plotly_chart(fig, use_container_width=True, config=config)

    with fig_col2:
        st.markdown("### Offensive Breakdown")
        fig = two_sided_barchart(red_offensive_stats, blue_offensive_stats)

        st.plotly_chart(fig, use_container_width=True, config=config)

    # Show stats table
    table_chart = stats_table(red_stats, blue_stats)
    st.plotly_chart(table_chart, use_container_width=True, config=config)

    st.markdown("### Detailed Data View")
    st.dataframe(
        df[
            [
                "event",
                "matchup",
                "red_name",
                "blue_name",
                "red_predicted_probability",
                "blue_predicted_probability",
                "prediction",
            ]
        ].style.pipe(prediction_table)
    )
