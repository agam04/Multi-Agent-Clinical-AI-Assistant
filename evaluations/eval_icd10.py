"""
ICD-10 extraction evaluation script.

Runs DiagnosticCoderAgent against the 100-sample ground-truth dataset and
reports per-sample and aggregate precision / recall / F1.

Usage:
    python -m evaluations.eval_icd10                  # full run
    python -m evaluations.eval_icd10 --limit 10       # first N samples only
    python -m evaluations.eval_icd10 --out results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

DATASET_PATH = Path(__file__).parent / "synthetic_icd10_dataset.json"
DEFAULT_OUT = Path(__file__).parent / "eval_results.json"


def _code_set(entries: List[Any]) -> set:
    """Collect normalised codes from either ground-truth dicts or DiagnosticCode objects."""
    codes = set()
    for e in entries:
        code = e.get("code") if isinstance(e, dict) else getattr(e, "code", None)
        if code:
            codes.add(code.strip().upper())
    return codes


def _category_set(codes: set) -> set:
    """Roll codes up to their 3-character ICD-10 category (the part before the dot).

    e.g. G43.909 → G43, K35.80 → K35, R51 → R51. Category-level scoring credits
    the model for identifying the right disease family even when it misses the
    finer specificity digits — the standard relaxed metric for ICD-10 coding.
    """
    return {c.split(".")[0] for c in codes}


def _prf(predicted: set, truth: set) -> Dict[str, float]:
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def run_evaluation(limit: int | None = None, out_path: Path = DEFAULT_OUT) -> Dict[str, Any]:
    from app.graph.schema import WorkflowState
    from app.agents.coder import DiagnosticCoderAgent

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    print(f"Loading DiagnosticCoderAgent...")
    agent = DiagnosticCoderAgent()
    print(f"Model loaded. Evaluating {len(dataset)} samples...\n")

    sample_results = []
    total_p = total_r = total_f1 = 0.0
    cat_total_p = cat_total_r = cat_total_f1 = 0.0
    exact_matches = 0
    total_latency = 0.0

    for i, sample in enumerate(dataset, 1):
        note = sample["note"]
        truth_codes = _code_set(sample.get("icd10_codes", []))

        state = WorkflowState(
            task="coding",
            payload={"note": note, "clinical_note": note},
            result=None,
            error=None,
        )

        t0 = time.perf_counter()
        output = agent.execute(state)
        elapsed = time.perf_counter() - t0
        total_latency += elapsed

        if output.error or not output.result:
            predicted_codes = set()
            error = output.error or "empty result"
        else:
            predicted_codes = _code_set(output.result)
            error = None

        metrics = _prf(predicted_codes, truth_codes)
        cat_metrics = _prf(_category_set(predicted_codes), _category_set(truth_codes))
        exact = predicted_codes == truth_codes

        if exact:
            exact_matches += 1
        total_p += metrics["precision"]
        total_r += metrics["recall"]
        total_f1 += metrics["f1"]
        cat_total_p += cat_metrics["precision"]
        cat_total_r += cat_metrics["recall"]
        cat_total_f1 += cat_metrics["f1"]

        sample_result = {
            "sample_id": i,
            "truth_codes": sorted(truth_codes),
            "predicted_codes": sorted(predicted_codes),
            "exact_match": exact,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "category_precision": round(cat_metrics["precision"], 4),
            "category_recall": round(cat_metrics["recall"], 4),
            "category_f1": round(cat_metrics["f1"], 4),
            "latency_s": round(elapsed, 2),
            "error": error,
        }
        sample_results.append(sample_result)

        status = "✓" if exact else "✗"
        print(
            f"[{i:3d}/{len(dataset)}] {status}  "
            f"exact F1={metrics['f1']:.2f}  cat F1={cat_metrics['f1']:.2f}  ({elapsed:.1f}s)"
        )

    n = len(dataset)
    aggregate = {
        "n_samples": n,
        "exact_match_rate": round(exact_matches / n, 4),
        "macro_precision": round(total_p / n, 4),
        "macro_recall": round(total_r / n, 4),
        "macro_f1": round(total_f1 / n, 4),
        "category_macro_precision": round(cat_total_p / n, 4),
        "category_macro_recall": round(cat_total_r / n, 4),
        "category_macro_f1": round(cat_total_f1 / n, 4),
        "avg_latency_s": round(total_latency / n, 2),
    }

    print("\n" + "=" * 56)
    print("AGGREGATE RESULTS")
    print("=" * 56)
    print(f"  Samples evaluated      : {n}")
    print(f"  Avg latency / sample   : {aggregate['avg_latency_s']:.2f} s")
    print(f"  Exact match rate       : {aggregate['exact_match_rate']:.1%}")
    print("  -- Exact code match --")
    print(f"  Macro precision        : {aggregate['macro_precision']:.3f}")
    print(f"  Macro recall           : {aggregate['macro_recall']:.3f}")
    print(f"  Macro F1               : {aggregate['macro_f1']:.3f}")
    print("  -- 3-char category match --")
    print(f"  Macro precision        : {aggregate['category_macro_precision']:.3f}")
    print(f"  Macro recall           : {aggregate['category_macro_recall']:.3f}")
    print(f"  Macro F1               : {aggregate['category_macro_f1']:.3f}")
    print("=" * 56)

    report = {"aggregate": aggregate, "samples": sample_results}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull results written to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICD-10 extraction")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, out_path=args.out)


if __name__ == "__main__":
    main()
