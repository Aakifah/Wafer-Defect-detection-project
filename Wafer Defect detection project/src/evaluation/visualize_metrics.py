"""Visualization script to compare the metrics of different model versions."""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_ORDER = [
    ("base_cnn", "1. Base CNN"),
    ("focal_cnn", "2. + Focal Loss"),
    ("bayesian_cnn", "3. + Bayesian Opt"),
    ("asvd_cnn", "4. + ASVD"),
    ("quantized_cnn", "5. + INT8 Quant")
]

def setup_style() -> None:
    """Applies modern, clean styling for Matplotlib/Seaborn."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_theme(style="whitegrid", palette="deep")
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.edgecolor": "#E0E0E0",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.alpha": 0.5,
        "grid.color": "#E0E0E0",
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.titlesize": 16,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

def visualize_metrics() -> None:
    """Aggregates and visualizes the metrics of all trained model versions."""
    eval_dir = Path("evaluation")
    img_dir = Path("images")
    img_dir.mkdir(parents=True, exist_ok=True)
    
    if not eval_dir.exists():
        logger.error("Evaluation directory not found.")
        return
        
    metrics_dict = {}
    
    for sub_dir in eval_dir.iterdir():
        if sub_dir.is_dir():
            metrics_path = sub_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                    metrics_dict[sub_dir.name] = {
                        "Micro F1": data.get("micro_f1", data.get("test_micro_f1", 0.0)),
                        "Macro F1": data.get("macro_f1", data.get("test_macro_f1", 0.0)),
                        "mAP": data.get("mAP", data.get("test_mAP", 0.0)),
                    }
                
    if not metrics_dict:
        logger.warning("No metrics found to visualize.")
        return
        
    ordered_data = []
    for dir_name, display_name in PIPELINE_ORDER:
        if dir_name in metrics_dict:
            entry = metrics_dict[dir_name].copy()
            entry["Stage"] = display_name
            ordered_data.append(entry)
            
    df = pd.DataFrame(ordered_data)
    
    if df.empty:
        logger.warning("Found metrics, but none matched the expected pipeline directories.")
        ordered_data = [{"Stage": k, **v} for k, v in metrics_dict.items()]
        df = pd.DataFrame(ordered_data)
    
    df.set_index("Stage", inplace=True)
    logger.info("\n=== Model Performance Comparison ===\n%s\n", df.to_string())
    df.to_csv(eval_dir / "comparison.csv")
    
    try:
        import matplotlib.pyplot as plt
        setup_style()
        
        metric_colors = {
            "mAP": "#FF007F",       
            "Macro F1": "#00B4D8",  
            "Micro F1": "#8A2BE2"  
        }


        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["mAP"], marker='o', linewidth=3.0, markersize=10, color=metric_colors["mAP"], label="mAP")
        plt.plot(df.index, df["Macro F1"], marker='s', linewidth=3.0, markersize=10, color=metric_colors["Macro F1"], label="Macro F1")
        
        plt.title("The Optimization Journey: Cumulative Improvements", pad=20, fontweight="bold", fontsize=16, color="#1A1A1A")
        plt.ylabel("Score", fontweight="bold")
        plt.xticks(rotation=15, fontweight="bold")
        plt.ylim(max(0, df.min().min() - 0.1), min(1.0, df.max().max() + 0.05))
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        plt.savefig(img_dir / "plot_a_journey.svg", format="svg", transparent=True, bbox_inches="tight")
        plt.close()

        base_focal_df = df[df.index.str.contains("Base|Focal", case=False)]
        if not base_focal_df.empty:
            ax = base_focal_df[["Macro F1", "Micro F1"]].plot(
                kind="bar", figsize=(8, 5), color=[metric_colors["Macro F1"], metric_colors["Micro F1"]], width=0.6, zorder=3
            )
            plt.title("The Imbalance Fix: Addressing Minority Classes", pad=20, fontweight="bold", fontsize=16, color="#1A1A1A")
            plt.ylabel("Score", fontweight="bold")
            plt.xticks(rotation=0, fontweight="bold")
            
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=11, fontweight="bold")
                            
            plt.ylim(0, 1.1)
            plt.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            plt.savefig(img_dir / "plot_b_imbalance.svg", format="svg", transparent=True, bbox_inches="tight")
            plt.close()

        peak_idx = next((idx for idx in df.index if "Bayesian" in idx), None)
        final_idx = next((idx for idx in df.index if "Quant" in idx), None)
        

        if peak_idx and final_idx:
            comp_df = df.loc[[peak_idx, final_idx]]
        elif len(df) >= 2:
            comp_df = df.iloc[[0, -1]]
        else:
            comp_df = pd.DataFrame()

        if not comp_df.empty:
            ax = comp_df[["mAP", "Macro F1"]].plot(
                kind="bar", figsize=(8, 5), color=[metric_colors["mAP"], metric_colors["Macro F1"]], width=0.6, zorder=3
            )
            plt.title("The Compression Trade-off: Peak Accuracy vs Edge Latency", pad=20, fontweight="bold", fontsize=16, color="#1A1A1A")
            plt.ylabel("Score", fontweight="bold")
            plt.xticks(rotation=0, fontweight="bold")
            
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=11, fontweight="bold")
                            
            plt.ylim(0, 1.1)
            plt.legend(loc='lower right', fontsize=12)
            plt.tight_layout()
            plt.savefig(img_dir / "plot_c_compression.svg", format="svg", transparent=True, bbox_inches="tight")
            plt.close()

        logger.info("Successfully generated SVGs in the 'images/' directory.")
        
    except ImportError as e:
        logger.error(f"Visualization dependencies missing.. Error: {e}")

if __name__ == "__main__":
    visualize_metrics()