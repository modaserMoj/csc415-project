"""
Print accuracy table from existing results/*.json (3 datasets: gsm8k, math, svamp).
Run from project root: python eval/print_results_table.py
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATASET_ORDER = ("gsm8k", "math", "svamp")
MODEL_ORDER = (
    "base_0.5b",
    "baseline_gsm8k_0.5b",
    "baseline_math_0.5b",
    "phase2_gsm8k_0.5b",
    "phase2_math_0.5b",
)

def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"No results dir: {RESULTS_DIR}")
        return

    # Collect (model, dataset) -> accuracy from existing files
    accs = {}
    for f in os.listdir(RESULTS_DIR):
        if not f.endswith(".json"):
            continue
        name = f[:-5]
        if "_" not in name:
            continue
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model_name, dataset_name = parts
        path = os.path.join(RESULTS_DIR, f)
        try:
            with open(path) as fp:
                d = json.load(fp)
            accs[(model_name, dataset_name)] = d.get("accuracy", 0) * 100
        except Exception:
            accs[(model_name, dataset_name)] = None

    # Use only datasets that appear in results
    datasets = [d for d in DATASET_ORDER if any(accs.get((m, d)) is not None for m in MODEL_ORDER)]
    if not datasets:
        datasets = sorted({d for (_, d) in accs})

    # Header
    col_width = 8
    header = "Model                  |" + "|".join(f" {d.upper():^{col_width}} " for d in datasets)
    sep = "-" * 23 + "|" + ("-" * (col_width + 2)) * len(datasets)
    print(header)
    print(sep)

    for model_name in MODEL_ORDER:
        row = f"{model_name:<23}|"
        for d in datasets:
            val = accs.get((model_name, d))
            if val is not None:
                cell = f"{val:.1f}%"
            else:
                cell = "-"
            row += f" {cell:^{col_width}} |"
        print(row)

if __name__ == "__main__":
    main()
