#!/usr/bin/env python
import plotext as plt
from obspy.core.event import Event


def plot_arrival_time(event: Event) -> None:
    origin = event.preferred_origin()

    P_arrival_time = []
    P_distance = []
    P_residual = []
    P_colors = []

    S_arrival_time = []
    S_distance = []
    S_residual = []
    S_colors = []

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
            P_arrival_time.append(pick.time - earliest_arrival_time)
            P_distance.append(arrival.distance * 111.1)
            P_residual.append(arrival.time_residual)
            if arrival.time_weight != 0:
                if arrival.phase == "Pn":
                    P_colors.append("red")
                elif arrival.phase == "Pg":
                    P_colors.append("red+")
                else:
                    P_colors.append("orange")
            else:
                P_colors.append("grey")

        elif "S" in arrival.phase:
            S_arrival_time.append(pick.time - earliest_arrival_time)
            S_distance.append(arrival.distance * 111.1)
            S_residual.append(arrival.time_residual)
            if arrival.time_weight != 0:
                if arrival.phase == "Sn":
                    S_colors.append("blue")
                elif arrival.phase == "Sg":
                    S_colors.append("blue+")
                else:
                    S_colors.append("cyan")
            else:
                S_colors.append("grey")

    # make 2 subplots side by side
    plt.subplots(1, 2)

    plt.subplot(1, 1)
    plt.scatter(P_distance, P_arrival_time, color=P_colors, marker="*", label="P|Pn|Pg")
    plt.scatter(S_distance, S_arrival_time, color=S_colors, label="S|Sn|Sg")
    plt.xlabel("Distance (km)")
    plt.ylabel("Arrival tim (s)")
    plt.title("Arrival time vs. distance")
    plt.plot_size(50, 25)
    plt.theme("pro")

    plt.subplot(1, 2)
    plt.hline(0, "white")
    plt.scatter(P_distance, P_residual, color=P_colors, marker="*", label="P|Pn|Pg")
    plt.scatter(S_distance, S_residual, color=S_colors, label="S|Sn|Sg")
    plt.xlabel("Distance (km)")
    plt.ylabel("Residual (s)")
    plt.title("Residual vs. distance")
    plt.plot_size(50, 25)
    plt.theme("pro")

    plt.show()
