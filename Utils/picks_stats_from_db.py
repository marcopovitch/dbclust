#!/usr/bin/env python3
import argparse
import sqlite3
import pandas as pd
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate statistics on picks")
    parser.add_argument("-d", "--db", help="Database", required=True)
    parser.add_argument(
        "-o", "--output_file", help="Output csv file with picks stats", required=True
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots interactively"
    )
    args = parser.parse_args()

    if os.path.exists(args.db):
        db_path = args.db
    else:
        print(f"File not found: {args.db}")
        exit()

    if os.path.exists(args.output_file):
        print(f"File already exists: {args.output_file}")
        exit()
    else:
        output_file = args.output_file

    # Derive base path for PNG files and report from output file
    output_base = os.path.splitext(output_file)[0]
    report_file = output_base + "_report.txt"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # SQL query to get pick probabilities for P and S phases
    query_probabilities = """
        SELECT 
            p.probability,
            p.phase_hint,
            e.event_id
        FROM 
            events e
        JOIN 
            origins o ON e.event_id = o.event_id
        JOIN 
            arrivals a ON o.id = a.origin_id
        JOIN 
            picks p ON a.pick_id = p.id
        WHERE 
            o.preferred = 1
            AND a.time_weight > 0
            AND p.evaluation_mode = 'automatic'
            AND (p.phase_hint LIKE 'P%' OR p.phase_hint LIKE 'S%')
            AND p.probability IS NOT NULL;
    """

    # SQL query to get average probability per event
    query_avg_prob_per_event = """
        SELECT 
            e.event_id,
            AVG(CASE WHEN p.phase_hint LIKE 'P%' THEN p.probability ELSE NULL END) as avg_p_prob,
            AVG(CASE WHEN p.phase_hint LIKE 'S%' THEN p.probability ELSE NULL END) as avg_s_prob,
            AVG(p.probability) as avg_probability
        FROM 
            events e
        JOIN 
            origins o ON e.event_id = o.event_id
        JOIN 
            arrivals a ON o.id = a.origin_id
        JOIN 
            picks p ON a.pick_id = p.id
        WHERE 
            o.preferred = 1
            AND a.time_weight > 0
            AND p.evaluation_mode = 'automatic'
            AND (p.phase_hint LIKE 'P%' OR p.phase_hint LIKE 'S%')
            AND p.probability IS NOT NULL
        GROUP BY 
            e.event_id
        HAVING 
            COUNT(p.id) > 0;
    """

    # SQL query for event statistics
    query = """
        SELECT
            e.event_id,
            e.nb_agencies,
            e.agency_names,
            e.agency_ai_contributors,
            SUM(CASE WHEN p.evaluation_mode = 'automatic' AND p.phase_hint LIKE 'P%' THEN 1 ELSE 0 END) AS automatic_picks_P,
            SUM(CASE WHEN p.evaluation_mode = 'automatic' AND p.phase_hint LIKE 'S%' THEN 1 ELSE 0 END) AS automatic_picks_S,
            SUM(CASE WHEN p.evaluation_mode = 'manual' THEN 1 ELSE 0 END) AS manual_picks,
            AVG(CASE WHEN p.evaluation_mode = 'automatic' AND p.phase_hint LIKE 'P%' THEN p.probability ELSE NULL END) AS avg_probability_P,
            AVG(CASE WHEN p.evaluation_mode = 'automatic' AND p.phase_hint LIKE 'S%' THEN p.probability ELSE NULL END) AS avg_probability_S
        FROM
            events e
        JOIN
            origins o ON e.event_id = o.event_id
        JOIN
            arrivals a ON o.id = a.origin_id
        JOIN
            picks p ON a.pick_id = p.id
        WHERE
            o.preferred = 1
            AND a.time_weight > 0
        GROUP BY
            e.event_id;
    """

    # Load results into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)
    print(df)
    df.to_csv(output_file, index=False)

    # Get probabilities data
    df_probs = pd.read_sql_query(query_probabilities, conn)
    df_avg_probs = pd.read_sql_query(query_avg_prob_per_event, conn)

    # Calculate statistics for P phases
    p_probs = df_probs[df_probs["phase_hint"].str.startswith("P")]["probability"]
    p_mean = p_probs.mean()
    p_std = p_probs.std()

    # Calculate statistics for S phases
    s_probs = df_probs[df_probs["phase_hint"].str.startswith("S")]["probability"]
    s_mean = s_probs.mean()
    s_std = s_probs.std()

    # Calculate statistics for average probability per event
    avg_prob_mean = df_avg_probs["avg_probability"].mean()
    avg_prob_std = df_avg_probs["avg_probability"].std()

    # Calculate averages for picks
    average_automatic_picks_P = df["automatic_picks_P"].mean()
    average_automatic_picks_S = df["automatic_picks_S"].mean()
    average_manual_picks = df["manual_picks"].mean()
    average_probability_P = df["avg_probability_P"].mean()
    average_probability_S = df["avg_probability_S"].mean()

    # Display results
    print(
        f"The average number of automatic P picks per event is: {average_automatic_picks_P:.2f}"
    )
    print(
        f"The average number of automatic S picks per event is: {average_automatic_picks_S:.2f}"
    )
    print(
        f"The average number of manual picks per event is: {average_manual_picks:.2f}"
    )

    total_picks = (
        average_manual_picks + average_automatic_picks_P + average_automatic_picks_S
    )
    # Calculate percentages
    percentage_manual_picks = average_manual_picks / total_picks * 100
    percentage_automatic_picks_P = average_automatic_picks_P / total_picks * 100
    percentage_automatic_picks_S = average_automatic_picks_S / total_picks * 100

    print(f"The average percentage of manual picks is: {percentage_manual_picks:.2f}%")
    print(
        f"The average percentage of automatic P picks is: {percentage_automatic_picks_P:.2f}%"
    )
    print(
        f"The average percentage of automatic S picks is: {percentage_automatic_picks_S:.2f}%"
    )

    # Plot the percentages
    labels = ["Manual", "Auto P", "Auto S"]
    counts = [
        average_manual_picks,
        average_automatic_picks_P,
        average_automatic_picks_S,
    ]
    sizes = [
        percentage_manual_picks,
        percentage_automatic_picks_P,
        percentage_automatic_picks_S,
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    explode = (0.05, 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _texts, _autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=140,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
        pctdistance=0.75,
        textprops=dict(fontsize=12, fontweight="bold", color="white"),
    )

    # Legend with counts and percentages
    legend_labels = [
        f"{lbl}  —  avg {cnt:.1f} picks/event  ({pct:.1f}%)"
        for lbl, cnt, pct in zip(labels, counts, sizes)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Pick type",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    ax.set_title("Distribution of pick types\n(per event, averages)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(output_base + "_pick_types.png", dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()

    print("\n=== Probabilities Statistics ===")
    print(f"P phases - Mean: {p_mean:.4f}, Std Dev: {p_std:.4f}")
    print(f"S phases - Mean: {s_mean:.4f}, Std Dev: {s_std:.4f}")

    # nb_agencies = 0 means no real operator agency contributed → pure AI event
    nb_automatic_events = len(df[df["nb_agencies"] == 0])
    print(f"\n{nb_automatic_events} events with only automatic picks (nb_agencies=0)")

    # Plot probability distributions
    plt.figure(figsize=(15, 5))

    # Bins: edges from 0.3 to 1.05 (step 0.1), ticks centered in each bar
    bin_edges = [round(i * 0.1, 1) for i in range(3, 12)]  # 0.3 … 1.1
    bin_centers = [round((bin_edges[i] + bin_edges[i + 1]) / 2, 2) for i in range(len(bin_edges) - 1)]
    tick_labels = [f"{round(b, 1):.1f}" for b in bin_edges[:-1]]

    # P phases histogram
    plt.subplot(1, 2, 1)
    plt.hist(p_probs, bins=bin_edges, color="lightcoral", edgecolor="black")
    plt.title("Probability Distribution - P Phases")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.xticks(bin_centers, tick_labels)
    plt.grid(True, alpha=0.3)

    # S phases histogram
    plt.subplot(1, 2, 2)
    plt.hist(s_probs, bins=bin_edges, color="lightskyblue", edgecolor="black")
    plt.title("Probability Distribution - S Phases")
    plt.xlabel("Probability")
    plt.xticks(bin_centers, tick_labels)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_base + "_prob_distributions.png", dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()

    # Plot average probability per event
    plt.figure(figsize=(12, 5))

    # Histogram of average probability per event
    plt.subplot(1, 2, 1)
    plt.hist(
        df_avg_probs["avg_probability"], bins=20, color="lightgreen", edgecolor="black"
    )
    plt.axvline(
        avg_prob_mean,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label=f"Moyenne: {avg_prob_mean:.2f}",
    )
    plt.axvline(
        avg_prob_mean + avg_prob_std,
        color="gray",
        linestyle="dashed",
        linewidth=0.8,
        label="±1 std dev",
    )
    plt.axvline(
        avg_prob_mean - avg_prob_std, color="gray", linestyle="dashed", linewidth=0.8
    )
    plt.title("Average Probability per Event")
    plt.xlabel("Average Probability")
    plt.ylabel("Number of Events")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Scatter plot with heatmap of P vs S average probabilities
    plt.subplot(1, 2, 2)

    # Create hexbin for heatmap with better colormap
    hb = plt.hexbin(
        df_avg_probs["avg_p_prob"],
        df_avg_probs["avg_s_prob"],
        gridsize=30,
        cmap="YlOrRd",  # Yellow-Orange-Red colormap for better visibility
        mincnt=1,
        bins="log",
        edgecolors="none",
        xscale='linear',
        yscale='linear'
    )
    
    # Add colorbar with better formatting
    cb = plt.colorbar(hb, label="Number of events (log scale)")
    # Update colorbar ticks to show actual counts
    cb.set_ticks([1, 10, 100, 1000] if len(df_avg_probs) > 1000 else [1, 10, 100])
    cb.set_ticklabels(['1', '10', '100', '1000+'] if len(df_avg_probs) > 1000 else ['1', '10', '100+'])

    # Add diagonal line
    plt.axline(
        (0.3, 0.3), (1.0, 1.0), color="red", linestyle="--", linewidth=1, label="y = x"
    )

    # Add mean point
    plt.scatter(
        [df_avg_probs["avg_p_prob"].mean()],
        [df_avg_probs["avg_s_prob"].mean()],
        color="red",
        s=100,
        marker="x",
        linewidth=2,
        label="Mean",
    )

    plt.xlabel("Average P Phase Probability")
    plt.ylabel("Average S Phase Probability")
    plt.title("P vs S - Average Probabilities per Event")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Ensure equal aspect ratio
    plt.gca().set_aspect("equal", adjustable="box")

    # Add correlation coefficient
    corr = df_avg_probs[["avg_p_prob", "avg_s_prob"]].corr().iloc[0, 1]
    plt.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(output_base + "_avg_prob_per_event.png", dpi=150, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close()

    # Print statistics
    print("\n=== Average Probability Statistics per Event ===")
    print(f"Average probability: {avg_prob_mean:.4f}")
    print(f"Standard deviation: {avg_prob_std:.4f}")
    print(f"Median: {df_avg_probs['avg_probability'].median():.4f}")
    print(f"Number of events: {len(df_avg_probs)}")

    # Generate text report
    import json as _json
    nb_events = len(df)
    total_auto_P = df["automatic_picks_P"].sum()
    total_auto_S = df["automatic_picks_S"].sum()
    total_manual = df["manual_picks"].sum()
    total_picks_all = total_auto_P + total_auto_S + total_manual

    # Agency statistics from events table
    nb_mixed_events = len(df[df["nb_agencies"] > 0])  # at least one real operator agency

    # Count contributions per operator agency
    agency_event_counts: dict = {}
    agency_ai_event_counts: dict = {}
    for _, row in df.iterrows():
        for agency in _json.loads(row["agency_names"] or "[]"):
            agency_event_counts[agency] = agency_event_counts.get(agency, 0) + 1
        for agency in _json.loads(row["agency_ai_contributors"] or "[]"):
            agency_ai_event_counts[agency] = agency_ai_event_counts.get(agency, 0) + 1

    agency_lines = []
    if agency_event_counts:
        agency_lines.append("  Operator agencies (events with real picks):")
        for ag, cnt in sorted(agency_event_counts.items(), key=lambda x: -x[1]):
            agency_lines.append(f"    {ag:<20} {cnt} events")
    if agency_ai_event_counts:
        agency_lines.append("  AI-only agencies (no real catalogue pick):")
        for ag, cnt in sorted(agency_ai_event_counts.items(), key=lambda x: -x[1]):
            agency_lines.append(f"    {ag:<20} {cnt} events")

    report_lines = [
        "=" * 50,
        "PICKS STATISTICS REPORT",
        f"Database: {db_path}",
        "=" * 50,
        "",
        "--- Events ---",
        f"Total events:                  {nb_events}",
        f"AI-only events (nb_agencies=0):{nb_automatic_events:>6}",
        f"Mixed events   (nb_agencies>0):{nb_mixed_events:>6}",
        "",
        "--- Agency contributions ---",
        *agency_lines,
        "",
        "--- Picks per event (averages) ---",
        f"Automatic P picks:  {average_automatic_picks_P:.2f}  ({percentage_automatic_picks_P:.1f}%)",
        f"Automatic S picks:  {average_automatic_picks_S:.2f}  ({percentage_automatic_picks_S:.1f}%)",
        f"Manual picks:       {average_manual_picks:.2f}  ({percentage_manual_picks:.1f}%)",
        "",
        "--- Total picks ---",
        f"Automatic P:  {int(total_auto_P)}",
        f"Automatic S:  {int(total_auto_S)}",
        f"Manual:       {int(total_manual)}",
        f"Total:        {int(total_picks_all)}",
        "",
        "--- Probability statistics ---",
        f"P phases  - mean: {p_mean:.4f}, std: {p_std:.4f}",
        f"S phases  - mean: {s_mean:.4f}, std: {s_std:.4f}",
        f"Per event - mean: {avg_prob_mean:.4f}, std: {avg_prob_std:.4f}, median: {df_avg_probs['avg_probability'].median():.4f}",
        f"P vs S correlation: {corr:.3f}",
        "",
        "--- Output files ---",
        f"CSV:                  {output_file}",
        f"Pick types plot:      {output_base}_pick_types.png",
        f"Prob distributions:   {output_base}_prob_distributions.png",
        f"Avg prob per event:   {output_base}_avg_prob_per_event.png",
        f"Report:               {report_file}",
        "=" * 50,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    with open(report_file, "w") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to {report_file}")

    # Close the connection
    conn.close()
