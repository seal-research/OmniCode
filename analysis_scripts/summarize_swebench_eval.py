#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Set


@dataclass
class Aggregate:
    total_reports: int = 0
    resolved: int = 0
    unresolved: int = 0

    def add(self, is_resolved: bool) -> None:
        self.total_reports += 1
        if is_resolved:
            self.resolved += 1
        else:
            self.unresolved += 1

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["resolve_rate"] = round(self.resolved / self.total_reports, 4) if self.total_reports else 0.0
        return d


def load_predictions(predictions_path: Path) -> Set[str]:
    ids: Set[str] = set()
    text = predictions_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return ids

    # Try JSONL first
    if "\n{" in text and not text.lstrip().startswith("["):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "instance_id" in obj:
                    ids.add(obj["instance_id"])
            except Exception:
                continue
        return ids

    # Fallback JSON list
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            for obj in arr:
                if isinstance(obj, dict) and "instance_id" in obj:
                    ids.add(obj["instance_id"])
    except Exception:
        pass
    return ids


def iter_reports(eval_dir: Path) -> Iterable[Path]:
    # Recursively find report.json files
    yield from eval_dir.rglob("report.json")


def get_model_from_path(report_path: Path) -> str:
    # Expected layout: <...>/<run_id>/<model>/<instance_id>/report.json
    # model = parents[2]
    try:
        return report_path.parents[2].name
    except Exception:
        return "unknown_model"


def get_instance_from_report(report_path: Path) -> tuple[str, bool]:
    # report.json format: { "<instance_id>": { ..., "resolved": true/false, ... } }
    try:
        data = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(data, dict) or not data:
            return ("", False)
        (instance_id, payload), = data.items()
        resolved = False
        if isinstance(payload, dict):
            # Prefer SWE-bench CodeReview key
            if "resolved" in payload and isinstance(payload["resolved"], bool):
                resolved = payload["resolved"]
            # Some modes may use different acceptance flag (e.g., TestGeneration)
            elif "Test_Accept" in payload and isinstance(payload["Test_Accept"], bool):
                resolved = payload["Test_Accept"]
        return (str(instance_id), bool(resolved))
    except Exception:
        return ("", False)


def summarize(eval_dir: Path, predictions_path: Path | None) -> Dict:
    per_model: dict[str, Aggregate] = defaultdict(Aggregate)
    global_agg = Aggregate()

    completed_ids: Set[str] = set()
    all_models: Set[str] = set()

    for report in iter_reports(eval_dir):
        model = get_model_from_path(report)
        all_models.add(model)
        instance_id, is_resolved = get_instance_from_report(report)
        if not instance_id:
            continue
        completed_ids.add(instance_id)
        per_model[model].add(is_resolved)
        global_agg.add(is_resolved)

    submitted_ids: Set[str] = set()
    if predictions_path and predictions_path.exists():
        submitted_ids = load_predictions(predictions_path)

    summary = {
        "eval_dir": str(eval_dir.resolve()),
        "models": sorted(m for m in all_models if m != "unknown_model") or sorted(all_models),
        "global": global_agg.to_dict(),
        "per_model": {m: agg.to_dict() for m, agg in sorted(per_model.items())},
        "completed_ids_count": len(completed_ids),
        "submitted_ids_count": len(submitted_ids) if submitted_ids else None,
        "missing_reports_count": int(len(submitted_ids - completed_ids)) if submitted_ids else None,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize SWE-bench evaluation results (global and per-model)")
    ap.add_argument("eval_dir", type=str, help=(
        "Path to the evaluation directory that contains <run_id>/<model>/<instance>/report.json, "
        "or directly the <run_id> directory. For your 'swebench_eval' folder, point here."
    ))
    ap.add_argument("--predictions", type=str, default=None, help="Optional path to predictions JSON/JSONL to compute completeness")
    ap.add_argument("--out", type=str, default=None, help="Optional output JSON path; defaults to <eval_dir>/global_summary.json")

    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Eval dir not found: {eval_dir}")

    predictions_path = Path(args.predictions) if args.predictions else None
    summary = summarize(eval_dir, predictions_path)

    # Print concise human-readable summary
    g = summary["global"]
    print("=== Global Summary ===")
    print(f"Eval dir: {summary['eval_dir']}")
    print(f"Models: {', '.join(summary['models']) if summary['models'] else 'n/a'}")
    print(f"Reports: {g['total_reports']}, Resolved: {g['resolved']}, Unresolved: {g['unresolved']}, Resolve@1: {g['resolve_rate']:.3f}")
    if summary["submitted_ids_count"] is not None:
        print(f"Submitted: {summary['submitted_ids_count']}, Completed: {summary['completed_ids_count']}, Missing reports: {summary['missing_reports_count']}")

    # Per-model breakdown
    if summary["per_model"]:
        print("\n=== Per-Model ===")
        for model, agg in summary["per_model"].items():
            print(f"- {model}: total={agg['total_reports']} resolved={agg['resolved']} unresolved={agg['unresolved']} rate={agg['resolve_rate']:.3f}")

    out_path = Path(args.out) if args.out else (eval_dir / "global_summary.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[+] Wrote global summary to {out_path}")


if __name__ == "__main__":
    main()


