"""
Summarize all results/*.json into a single table file.

Run from project root:
    python eval/summarize_results.py

It creates:
    results/summary.txt
"""

import json
import os
from collections import defaultdict, OrderedDict


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "summary.txt")

# Nicer display names for models in the table.
MODEL_DISPLAY = {
    "base_0.5b": "qwen0.5b_base",
    "baseline_gsm8k_0.5b": "qwen0.5b_baseline_gsm8k",
    "baseline_math_0.5b": "qwen0.5b_baseline_math",
    "phase2_gsm8k_0.5b": "qwen0.5b_phase1_to_gsm8k",
    "phase2_math_0.5b": "qwen0.5b_phase1_to_math",
}

def main() -> None:
    if not os.path.isdir(RESULTS_DIR):
        print(f"No results dir: {RESULTS_DIR}")
        return

    # (model, dataset) -> accuracy
    accs = {}
    models = set()
    datasets = set()

    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue

        # If file name looks like model_dataset.json, prefer that.
        name = fname[:-5]
        if "_" in name:
            model_name, dataset_name = name.rsplit("_", 1)
        else:
            model_name = str(data.get("model", name))
            dataset_name = os.path.basename(str(data.get("dataset", "")))

        acc = float(data.get("accuracy", 0.0)) * 100
        accs[(model_name, dataset_name)] = acc
        models.add(model_name)
        datasets.add(dataset_name)

    if not accs:
        print("No result JSONs found.")
        return

    models = sorted(models)
    datasets = sorted(datasets)

    # Write table.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as out:
        header = "Model".ljust(26) + " | " + " | ".join(f"{d:^10}" for d in datasets)
        sep = "-" * len(header)
        out.write(header + "\n")
        out.write(sep + "\n")
        for m in models:
            display = MODEL_DISPLAY.get(m, m)
            row = display.ljust(26) + " | "
            cells = []
            for d in datasets:
                acc = accs.get((m, d))
                cells.append(f"{acc:5.1f}%" if acc is not None else "   -   ")
            row += " | ".join(f"{c:^10}" for c in cells)
            out.write(row + "\n")

    print(f"Wrote summary to {OUT_PATH}")


if __name__ == "__main__":
    main()

