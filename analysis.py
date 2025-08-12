import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def get_question_type(q_idx: int) -> str:
    """Categorizes questions by type based on their index."""
    if q_idx < 3:
        return "Vector Search"
    elif q_idx < 6:
        return "Graph Search"
    elif q_idx < 9:
        return "Hybrid Search"
    else:
        return "Complex Analysis"


def analyze_and_visualize_results(results_data):
    """Analyzes and visualizes the RAG pipeline comparison results."""

    print("\nGENERATING COMPREHENSIVE ANALYSIS...")

    # Convert results to pandas DataFrame for easier analysis
    analysis_data = []
    for pipeline in results_data["pipelines"]:
        for q_idx in range(len(results_data["questions"])):
            if q_idx in results_data["timing"][pipeline]:
                analysis_data.append(
                    {
                        "Pipeline": pipeline,
                        "Question_Index": q_idx,
                        "Question_Type": get_question_type(q_idx),
                        "Timing": results_data["timing"][pipeline][q_idx],
                        "Quality_Score": results_data["quality_scores"][pipeline][
                            q_idx
                        ],
                        "Answer_Length": len(
                            str(results_data["answers"][pipeline].get(q_idx, ""))
                        ),
                    }
                )

    df = pd.DataFrame(analysis_data)

    # 1. Performance Comparison
    print("\nPERFORMANCE COMPARISON:")
    performance_summary = (
        df.groupby("Pipeline")
        .agg(
            {
                "Timing": ["mean", "std", "min", "max"],
                "Quality_Score": ["mean", "std", "min", "max"],
                "Answer_Length": ["mean", "std"],
            }
        )
        .round(3)
    )
    print(performance_summary)

    # 2. Question Type Analysis
    print("\nQUESTION TYPE ANALYSIS:")
    question_analysis = (
        df.groupby(["Pipeline", "Question_Type"])
        .agg({"Timing": "mean", "Quality_Score": "mean"})
        .round(3)
    )
    print(question_analysis)

    # 2.5 Statistical Significance Tests (Wilcoxon signed-rank on Quality_Score)
    print("\nSIGNIFICANCE TESTS (Wilcoxon signed-rank, Quality_Score):")
    pipelines = results_data["pipelines"]
    for i in range(len(pipelines)):
        for j in range(i + 1, len(pipelines)):
            p1, p2 = pipelines[i], pipelines[j]
            # Align scores by question index to ensure paired comparison
            merged = (
                df[df["Pipeline"].isin([p1, p2])]
                .pivot(
                    index="Question_Index", columns="Pipeline", values="Quality_Score"
                )
                .dropna()
            )
            if not merged.empty:
                try:
                    stat, p_val = wilcoxon(merged[p1], merged[p2])
                    print(f"  {p1} vs {p2}: p-value = {p_val:.4f}")
                except ValueError:
                    # Raised when the two distributions are identical or sample size too small
                    print(
                        f"  {p1} vs {p2}: Wilcoxon test not applicable (identical samples)"
                    )
            else:
                print(f"  {p1} vs {p2}: insufficient paired data")

    # 3. Generate Visualizations
    print("\nGENERATING VISUALIZATIONS...")

    # Set up the plotting style
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("RAG Pipeline Comparison Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Average Response Time by Pipeline
    avg_timing = df.groupby("Pipeline")["Timing"].mean().sort_values(ascending=True)
    axes[0, 0].bar(
        avg_timing.index,
        avg_timing.values,
        color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
    )
    axes[0, 0].set_title("Average Response Time by Pipeline", fontweight="bold")
    axes[0, 0].set_ylabel("Time (seconds)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Plot 2: Average Quality Score by Pipeline
    avg_quality = (
        df.groupby("Pipeline")["Quality_Score"].mean().sort_values(ascending=False)
    )
    axes[0, 1].bar(
        avg_quality.index,
        avg_quality.values,
        color=["#96CEB4", "#45B7D1", "#4ECDC4", "#FF6B6B"],
    )
    axes[0, 1].set_title("Average Quality Score by Pipeline", fontweight="bold")
    axes[0, 1].set_ylabel("Quality Score")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Plot 3: Performance by Question Type
    question_performance = (
        df.groupby(["Question_Type", "Pipeline"])["Quality_Score"].mean().unstack()
    )
    question_performance.plot(
        kind="bar", ax=axes[1, 0], color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    )
    axes[1, 0].set_title(
        "Quality Score by Question Type and Pipeline", fontweight="bold"
    )
    axes[1, 0].set_ylabel("Quality Score")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].legend(title="Pipeline", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 4: Timing vs Quality Scatter
    for pipeline in results_data["pipelines"]:
        pipeline_data = df[df["Pipeline"] == pipeline]
        axes[1, 1].scatter(
            pipeline_data["Timing"],
            pipeline_data["Quality_Score"],
            label=pipeline,
            alpha=0.7,
            s=50,
        )

    axes[1, 1].set_title("Timing vs Quality Score", fontweight="bold")
    axes[1, 1].set_xlabel("Response Time (seconds)")
    axes[1, 1].set_ylabel("Quality Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the visualization
    plt.savefig("rag_pipeline_comparison.png", dpi=300, bbox_inches="tight")
    print("Visualization saved as 'rag_pipeline_comparison.png'")

    # Save each subplot as its own image
    subplot_filenames = [
        "plot_avg_response_time.png",
        "plot_avg_quality_score.png",
        "plot_quality_by_question_type.png",
        "plot_timing_vs_quality.png",
    ]

    # Ensure the figure is rendered before extracting subplots
    fig.canvas.draw()
    for ax, fname in zip(axes.flatten(), subplot_filenames):
        # Get bounding box of the axis relative to the figure and save just that area
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted()
        )
        fig.savefig(fname, dpi=300, bbox_inches=extent)

    print("Saved individual subplot images:", ", ".join(subplot_filenames))

    # 4. Detailed Analysis Report
    print("\nDETAILED ANALYSIS REPORT:")
    print("=" * 60)

    # Best performing pipeline overall
    best_overall = df.groupby("Pipeline")["Quality_Score"].mean().idxmax()
    print(f"Best Overall Pipeline: {best_overall}")

    # Fastest pipeline
    fastest = df.groupby("Pipeline")["Timing"].mean().idxmin()
    print(f"Fastest Pipeline: {fastest}")

    # Best for different question types
    for q_type in df["Question_Type"].unique():
        q_type_data = df[df["Question_Type"] == q_type]
        best_for_type = q_type_data.groupby("Pipeline")["Quality_Score"].mean().idxmax()
        print(f"Best for {q_type}: {best_for_type}")

    # 5. Recommendations
    print("\nRECOMMENDATIONS:")
    print("=" * 60)

    # Analyze strengths and weaknesses
    for pipeline in results_data["pipelines"]:
        pipeline_data = df[df["Pipeline"] == pipeline]
        avg_time = pipeline_data["Timing"].mean()
        avg_quality = pipeline_data["Quality_Score"].mean()

        print(f"\n{pipeline}:")
        if avg_time < 10:
            print(f"  Fast response time: {avg_time:.2f}s")
        else:
            print(f"  Slow response time: {avg_time:.2f}s")

        if avg_quality > 50:
            print(f"  High quality responses: {avg_quality:.1f}/100")
        else:
            print(f"  Lower quality responses: {avg_quality:.1f}/100")

    # 6. Save detailed results to CSV
    results_df = pd.DataFrame(analysis_data)
    results_df.to_csv("rag_pipeline_results.csv", index=False)
    print("\nDetailed results saved to 'rag_pipeline_results.csv'")

    return df
