"""
Inspect raw ChilleD/MultiArith examples from HuggingFace to find the answer field.

Run:
    python eval/inspect_multiarith_raw.py
"""

from datasets import load_dataset


def main() -> None:
    ds = load_dataset("ChilleD/MultiArith")
    print(ds)
    for i in range(3):
        ex = ds["train"][i]
        print(f"\n=== Raw example {i} ===")
        print("keys:", list(ex.keys()))
        for k, v in ex.items():
            print(f"{k!r}: {v!r}")


if __name__ == "__main__":
    main()

