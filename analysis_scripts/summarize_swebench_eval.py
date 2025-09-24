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

    def add_counts(self, resolved_count: int, unresolved_count: int) -> None:
        self.total_reports += resolved_count + unresolved_count
        self.resolved += resolved_count
        self.unresolved += unresolved_count

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


def summarize(eval_dir: Path, predictions_path: Path | None, model_filter: str | None) -> Dict:
    per_model: dict[str, Aggregate] = defaultdict(Aggregate)
    global_agg = Aggregate()

    completed_ids: Set[str] = set()
    all_models: Set[str] = set()

    for report in iter_reports(eval_dir):
        model = get_model_from_path(report)
        if model_filter and model_filter not in model:
            continue
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
        "model_filter": model_filter,
        "models": sorted(m for m in all_models if m != "unknown_model") or sorted(all_models),
        "global": global_agg.to_dict(),
        "per_model": {m: agg.to_dict() for m, agg in sorted(per_model.items())},
        "completed_ids_count": len(completed_ids),
        "submitted_ids_count": len(submitted_ids) if submitted_ids else None,
        "missing_reports_count": int(len(submitted_ids - completed_ids)) if submitted_ids else None,
    }
    return summary


# ---------------- Flat-file swebench_eval (per-instance JSONs) ---------------- #

def iter_flat_jsons(root: Path) -> Iterable[Path]:
    for p in root.glob("*.json"):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        # Ignore any summary files to avoid contaminating counts
        if name_lower.endswith("global_summary.json") or "summary" in name_lower:
            continue
        yield p


def infer_model_from_filename(name: str) -> str:
    # Example: openrouter__anthropic__claude-sonnet-4.claude_sonnet_acr_eval_codereview_pytest-dev__pytest-10081.json
    # Model token is everything before the first '.'
    return name.split(".", 1)[0]


def infer_mode_from_filename(name: str) -> str | None:
    if "codereview" in name:
        return "codereview"
    if "bugfixing" in name or "bugfix" in name:
        return "bugfixing"
    if "stylereview" in name:
        return "stylereview"
    if "testgen" in name or "gentests" in name:
        return "testgen"
    return None


def extract_resolved_from_arbitrary_json(path: Path) -> tuple[str, bool]:
    """Return (instance_id, resolved) best-effort from loosely structured JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ("", False)

    # 1) SWE-bench report.json shape: { instance_id: { resolved: bool, ... } }
    if isinstance(data, dict) and data:
        if "resolved" in data and isinstance(data["resolved"], bool):
            # Top-level resolved (rare)
            inst = data.get("instance_id", "") if isinstance(data.get("instance_id"), str) else ""
            return (inst, bool(data["resolved"]))
        if len(data) == 1:
            (k, v), = data.items()
            if isinstance(v, dict):
                if "resolved" in v and isinstance(v["resolved"], bool):
                    return (str(k), bool(v["resolved"]))
                if "Test_Accept" in v and isinstance(v["Test_Accept"], bool):
                    return (str(k), bool(v["Test_Accept"]))

    # 2) ACR per-instance outputs may be arbitrary; try common shapes
    if isinstance(data, dict):
        if "result" in data and isinstance(data["result"], dict):
            res = data["result"].get("resolved")
            if isinstance(res, bool):
                return (str(data.get("instance_id", "")), bool(res))
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            res = data["evaluation"].get("resolved")
            if isinstance(res, bool):
                return (str(data.get("instance_id", "")), bool(res))
        # test generation acceptance
        if "Test_Accept" in data and isinstance(data["Test_Accept"], bool):
            return (str(data.get("instance_id", "")), bool(data["Test_Accept"]))

    return ("", False)


def summarize_flat(eval_dir: Path, predictions_path: Path | None, mode_filter: str | None, model_filter: str | None) -> Dict:
    per_model: dict[str, Aggregate] = defaultdict(Aggregate)
    per_mode: dict[str, Aggregate] = defaultdict(Aggregate)
    global_agg = Aggregate()

    all_models: Set[str] = set()
    considered_files = 0

    for jf in iter_flat_jsons(eval_dir):
        fname = jf.name
        mode = infer_mode_from_filename(fname) or "unknown"
        if mode_filter and mode != mode_filter:
            continue
        model = infer_model_from_filename(fname)
        if model_filter and model_filter not in model:
            continue
        considered_files += 1
        all_models.add(model)

        # Attempt to parse as aggregated run-level report first
        is_counted = False
        try:
            data = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict) and (
                ("resolved_ids" in data and isinstance(data.get("resolved_ids"), list)) or
                ("unresolved_ids" in data and isinstance(data.get("unresolved_ids"), list))
            ):
                r_cnt = len(data.get("resolved_ids", []) or [])
                u_cnt = len(data.get("unresolved_ids", []) or [])
                # If neither list exists as non-empty, but there are completed_ids, infer unresolved
                if r_cnt == 0 and u_cnt == 0:
                    c_ids = data.get("completed_ids", []) or []
                    u_cnt = len(c_ids)
                per_model[model].add_counts(r_cnt, u_cnt)
                per_mode[mode].add_counts(r_cnt, u_cnt)
                global_agg.add_counts(r_cnt, u_cnt)
                is_counted = True
        except Exception:
            pass

        if is_counted:
            continue

        # Fallback: treat as single-instance style report
        _, is_resolved = extract_resolved_from_arbitrary_json(jf)
        per_model[model].add(is_resolved)
        per_mode[mode].add(is_resolved)
        global_agg.add(is_resolved)

    submitted_ids: Set[str] = set()
    if predictions_path and predictions_path.exists():
        submitted_ids = load_predictions(predictions_path)

    summary = {
        "eval_dir": str(eval_dir.resolve()),
        "mode_filter": mode_filter,
        "model_filter": model_filter,
        "models": sorted(all_models),
        "files_count": considered_files,
        "global": global_agg.to_dict(),
        "per_model": {m: agg.to_dict() for m, agg in sorted(per_model.items())},
        "per_mode": {m: agg.to_dict() for m, agg in sorted(per_mode.items())},
        "submitted_ids_count": len(submitted_ids) if submitted_ids else None,
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
    ap.add_argument("--mode", type=str, default=None, choices=["codereview", "bugfixing", "stylereview", "testgen"], help="Filter flat swebench_eval files by mode substring")
    ap.add_argument("--model-substr", type=str, default=None, help="Filter by model substring (works for both flat and nested layouts)")

    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"Eval dir not found: {eval_dir}")

    predictions_path = Path(args.predictions) if args.predictions else None
    # Decide mode: nested report.json vs flat per-instance JSONs
    has_reports = any(iter_reports(eval_dir))
    if has_reports:
        summary = summarize(eval_dir, predictions_path, args.model_substr)
    else:
        summary = summarize_flat(eval_dir, predictions_path, args.mode, args.model_substr)

    # Print concise human-readable summary
    g = summary["global"]
    print("=== Global Summary ===")
    print(f"Eval dir: {summary['eval_dir']}")
    if summary.get("mode_filter"):
        print(f"Mode filter: {summary['mode_filter']}")
    if summary.get("model_filter"):
        print(f"Model filter: {summary['model_filter']}")
    print(f"Models: {', '.join(summary.get('models', [])) if summary.get('models') else 'n/a'}")
    if "files_count" in summary:
        print(f"Files: {summary['files_count']}")
    print(f"Total: {g['total_reports']}, Resolved: {g['resolved']}, Unresolved: {g['unresolved']}, Resolve@1: {g['resolve_rate']:.3f}")
    if summary.get("submitted_ids_count") is not None and summary.get("completed_ids_count") is not None:
        print(f"Submitted: {summary['submitted_ids_count']}, Completed: {summary['completed_ids_count']}, Missing reports: {summary['missing_reports_count']}")

    # Per-model breakdown
    if summary.get("per_model"):
        print("\n=== Per-Model ===")
        for model, agg in summary["per_model"].items():
            print(f"- {model}: total={agg['total_reports']} resolved={agg['resolved']} unresolved={agg['unresolved']} rate={agg['resolve_rate']:.3f}")
    if summary.get("per_mode"):
        print("\n=== Per-Mode ===")
        for mode, agg in summary["per_mode"].items():
            print(f"- {mode}: total={agg['total_reports']} resolved={agg['resolved']} unresolved={agg['unresolved']} rate={agg['resolve_rate']:.3f}")

    out_path = Path(args.out) if args.out else (eval_dir / "global_summary.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[+] Wrote global summary to {out_path}")


if __name__ == "__main__":
    main()


