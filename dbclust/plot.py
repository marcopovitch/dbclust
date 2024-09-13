#!/usr/bin/env python
import math

import pandas as pd
import plotext as plt
import plotly.graph_objects as go
from icecream import ic
from obspy.core.event import Event
from plotly.subplots import make_subplots


def plot_arrival_time(
    event: Event,
    event_name: str = None,
    use_plotly: bool = True,
    df_polygons=None,  # pick delimiter polygons for that region
) -> None:

    origin = event.preferred_origin()
    ic(df_polygons)

    P_station_name = []
    P_arrival_time = []
    P_distance = []
    P_residual = []
    P_colors = []
    P_phase = []

    S_station_name = []
    S_arrival_time = []
    S_distance = []
    S_residual = []
    S_colors = []
    S_phase = []

    tolerance = 0.001

    earliest_arrival_time = None
    for arrival in origin.arrivals:
        pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
        if not pick:
            continue
        if earliest_arrival_time is None:
            earliest_arrival_time = pick.time
        elif pick.time < earliest_arrival_time:
            earliest_arrival_time = pick.time

    for arrival in origin.arrivals:
        pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
        if not pick:
            continue

        # append the arrival time with respect to the earliest arrival time for P phase
        if "P" in arrival.phase:
            P_station_name.append(pick.waveform_id.get_seed_string())
            P_phase.append(arrival.phase)
            P_arrival_time.append(pick.time - earliest_arrival_time)
            P_distance.append(arrival.distance * 111.1)
            P_residual.append(arrival.time_residual)
            if abs(arrival.time_weight) > tolerance:
                if arrival.phase == "Pg":
                    P_colors.append("red")
                elif arrival.phase == "Pn":
                    P_colors.append("red")
                else:
                    P_colors.append("orange")
            else:
                # print(pick)
                P_colors.append("lightgray")

        elif "S" in arrival.phase:
            S_station_name.append(pick.waveform_id.get_seed_string())
            S_phase.append(arrival.phase)
            S_arrival_time.append(pick.time - earliest_arrival_time)
            S_distance.append(arrival.distance * 111.1)
            S_residual.append(arrival.time_residual)
            if abs(arrival.time_weight) > tolerance:
                if arrival.phase == "Sg":
                    S_colors.append("green")
                elif arrival.phase == "Sn":
                    S_colors.append("green")
                else:
                    S_colors.append("cyan")
            else:
                # print(pick)
                S_colors.append("lightgray")

    P_df = pd.DataFrame(
        {
            "station": P_station_name,
            "P_arrival_time": P_arrival_time,
            "P_distance": P_distance,
            "P_residual": P_residual,
            "P_colors": P_colors,
            "P_phase": P_phase,
        }
    )

    S_df = pd.DataFrame(
        {
            "station": S_station_name,
            "S_arrival_time": S_arrival_time,
            "S_distance": S_distance,
            "S_residual": S_residual,
            "S_colors": S_colors,
            "S_phase": S_phase,
        }
    )

    if use_plotly:
        make_plot_with_plotly(P_df, S_df, event_name, df_polygons)
    else:
        make_plot_with_plotext(P_df, S_df, event_name)


def make_plot_with_plotly(
    P_df: pd.DataFrame, S_df: pd.DataFrame, event_name: str = None, df_polygons=None
) -> None:
    """
    Generate a plot using Plotly library to visualize the arrival time vs. distance and residual vs. distance
    for P and S phases on the same plot.
    Args:
        P_df (pd.DataFrame): DataFrame containing P phase data with columns 'distance', 'arrival_time', and 'colors'.
        S_df (pd.DataFrame): DataFrame containing S phase data with columns 'distance', 'arrival_time', and 'colors'.
        event_name (str, optional): Name of the event. Defaults to None.
        df_polygons (pd.DataFrame, optional): DataFrame containing the polygons to plot. Defaults to None.
    Returns:
        None
    """

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Arrival time vs. distance", "Residual vs. distance"),
    )

    ###############
    # Graphic 1 #
    ###############

    # Iterate over all polygons in the DataFrame and plot them
    for idx, row in df_polygons.iterrows():
        polygon = row["geometry"]
        x, y = polygon.exterior.xy
        # convert x to km
        x = [i * 111.1 for i in x]
        x_list = list(x)
        y_list = list(y)
        fig.add_trace(
            go.Scatter(
                x=x_list, y=y_list, fill="toself", name=f"{row['name']} {row['region']}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=P_df["P_distance"],
            y=P_df["P_arrival_time"],
            mode="markers",
            marker=dict(
                size=10,
                symbol="circle",
                color=P_df["P_colors"],
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name="P phase",
            hovertemplate=(
                "%{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "Distance: %{x} km<br>"
                "Arrival time: %{y} s<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(P_df["station"], P_df["P_phase"])),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=S_df["S_distance"],
            y=S_df["S_arrival_time"],
            mode="markers",
            marker=dict(
                color=S_df["S_colors"],
                size=10,
                symbol="square",
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name="S phase",
            hovertemplate=(
                "%{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "Distance: %{x} km<br>"
                "Arrival time: %{y} s<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(S_df["station"], S_df["S_phase"])),
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
    fig.update_yaxes(title_text="Arrival time (s)", row=1, col=1)
    fig.update_xaxes(
        range=[0, math.ceil(max(P_df["P_distance"] + S_df["S_distance"]))], row=1, col=1
    )
    ymax = math.ceil(max(P_df["P_arrival_time"] + S_df["S_arrival_time"]))
    fig.update_yaxes(
        range=[0, ymax],
        row=1,
        col=1,
    )

    ###############
    # Graphic 2 #
    ###############
    fig.add_trace(
        go.Scatter(
            x=P_df["P_distance"],
            y=P_df["P_residual"],
            mode="markers",
            marker=dict(
                size=10,
                symbol="circle",
                color=P_df["P_colors"],
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name="P phase",
            hovertemplate=(
                "%{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "Distance: %{x} km<br>"
                "Arrival time: %{y} s<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(P_df["station"], P_df["P_phase"])),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=S_df["S_distance"],
            y=S_df["S_residual"],
            mode="markers",
            marker=dict(
                color=S_df["S_colors"],
                size=10,
                symbol="square",
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name="S phase",
            hovertemplate=(
                "%{customdata[0]}<br>"
                "%{customdata[1]}<br>"
                "Distance: %{x} km<br>"
                "Arrival time: %{y} s<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(S_df["station"], S_df["S_phase"])),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Distance (km)", row=1, col=2)
    fig.update_yaxes(title_text="Residual (s)", row=1, col=2)
    fig.update_xaxes(
        range=[0, math.ceil(max(P_df["P_distance"] + S_df["S_distance"]))], row=1, col=2
    )
    ymax = max(P_df["P_residual"] + S_df["S_residual"])
    ymin = min(P_df["P_residual"] + S_df["S_residual"])
    absmax = max(abs(ymax), abs(ymin))
    fig.update_yaxes(range=[-1.0 * absmax, absmax], row=1, col=2)

    if event_name:
        fig.update_layout(
            title=f"event: {event_name}",
        )

    fig.show()


def make_plot_with_plotext(
    P_df: pd.DataFrame, S_df: pd.DataFrame, event_name: str = None
) -> None:
    """
    Generate a plot using Plotext library to visualize the arrival time vs. distance and residual vs. distance
    for P and S phases on the same plot.
    Args:
        P_df (pd.DataFrame): DataFrame containing P phase data with columns 'distance', 'arrival_time', and 'colors'.
        S_df (pd.DataFrame): DataFrame containing S phase data with columns 'distance', 'arrival_time', and 'colors'.
        event_name (str, optional): Name of the event. Defaults to None.
    Returns:
        None
    Comment:
        The plotext library does not handle correctly the color of the markers (random behavior).
    """

    ic(P_df["P_colors"])
    ic(S_df["S_colors"])

    # make 2 subplots side by side
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.xlim(0, math.ceil(max(P_df["P_distance"] + S_df["S_distance"])))
    plt.scatter(
        P_df["P_distance"],
        P_df["P_arrival_time"],
        color=list(P_df["P_colors"]),
        marker="*",
        label="P|Pn|Pg",
    )
    plt.scatter(
        S_df["S_distance"],
        S_df["S_arrival_time"],
        color=list(S_df["S_colors"]),
        label="S|Sn|Sg",
    )
    plt.xlabel("Distance (km)")
    plt.ylabel("Arrival tim (s)")
    plt.title("Arrival time vs. distance")
    plt.plot_size(50, 25)
    plt.theme("pro")

    plt.subplot(1, 2)
    plt.hline(0, "white")
    plt.xlim(0, math.ceil(max(P_df["P_distance"] + S_df["S_distance"])))
    ymax = max(P_df["P_residual"] + S_df["S_residual"])
    ymin = min(P_df["P_residual"] + S_df["S_residual"])
    absmax = max(abs(ymax), abs(ymin))
    plt.ylim(-absmax, absmax)
    plt.scatter(
        P_df["P_distance"],
        P_df["P_residual"],
        color=list(P_df["P_colors"]),
        marker="*",
        label="P|Pn|Pg",
    )
    plt.scatter(
        S_df["S_distance"],
        S_df["S_residual"],
        color=list(S_df["S_colors"]),
        label="S|Sn|Sg",
    )
    plt.xlabel("Distance (km)")
    plt.ylabel("Residual (s)")
    plt.title("Residual vs. distance")
    plt.plot_size(50, 25)
    plt.theme("pro")
    plt.show()


if __name__ == "__main__":
    import os
    import sys
    import argparse
    from obspy import read_events

    # Use argparse to get the event file
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--event_file", help="Event file")
    parser.add_argument(
        "-p", "--plotly", help="Use Plotly", action="store_true", default=False
    )
    args = parser.parse_args()

    # check if the event file is provided
    if not args.event_file:
        parser.print_help()
        sys.exit(255)

    # check if the event file exists
    if not os.path.exists(args.event_file):
        print("The event file does not exist")
        sys.exit(1)

    cat = read_events(args.event_file)
    if len(cat) == 0:
        print("No event found in the event file")
        sys.exit(1)
    if len(cat) > 1:
        print("More than one event found in the event file")
        sys.exit(1)

    event_name = os.path.basename(args.event_file)
    event = cat[0]

    plot_arrival_time(event, event_name, use_plotly=args.plotly)
