#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def is_target_file(p: Path, model_token: str, run_prefix: str) -> bool:
    if not p.is_file() or not p.name.endswith('.json'):
        return False
    name = p.name
    # Only Claude CodeReview run-level reports of the form:
    #   <model_token>.<run_prefix><sanitized_instance>.json
    # e.g., openrouter__anthropic__claude-sonnet-4.claude_sonnet_acr_eval_codereview_pytest-dev__pytest-10081.json
    if model_token not in name or run_prefix not in name:
        return False
    # Exclude summaries
    lower = name.lower()
    if 'summary' in lower:
        return False
    return True


def count_from_report(path: Path) -> tuple[int, int]:
    """Return (resolved_count, unresolved_count) from a run-level report file.
    Falls back to treating all completed_ids as unresolved if resolved/unresolved lists missing.
    """
    try:
        data = json.loads(path.read_text(encoding='utf-8', errors='ignore'))
    except Exception:
        return (0, 0)

    if not isinstance(data, dict):
        return (0, 0)

    # Case A: Top-level boolean resolved field
    if isinstance(data.get('resolved'), bool):
        return (1, 0) if data['resolved'] else (0, 1)

    # Case B: SWE-bench per-instance report.json shape: { "<instance_id>": { resolved: bool, ... } }
    if len(data) == 1:
        try:
            (k, v), = data.items()
            if isinstance(v, dict):
                if isinstance(v.get('resolved'), bool):
                    return (1, 0) if v['resolved'] else (0, 1)
                if isinstance(v.get('Test_Accept'), bool):
                    return (1, 0) if v['Test_Accept'] else (0, 1)
        except Exception:
            pass

    # Case C: Run-level lists of IDs
    resolved = data.get('resolved_ids') or data.get('successful_ids')
    unresolved = data.get('unresolved_ids') or data.get('unsuccessful_ids')
    completed = data.get('completed_ids')

    r = len(resolved) if isinstance(resolved, list) else 0
    u = len(unresolved) if isinstance(unresolved, list) else 0

    # Case D: Run-level numeric counters
    if r == 0 and u == 0:
        for rk, uk in (
            ('resolved_instances', 'unresolved_instances'),
            ('pylint_success', 'pylint_failure'),  # not CodeReview, but harmless
        ):
            rv = data.get(rk)
            uv = data.get(uk)
            if isinstance(rv, int) and isinstance(uv, int):
                return (rv, uv)

    if r == 0 and u == 0 and isinstance(completed, list):
        # If lists are missing, conservatively count all completed as unresolved
        u = len(completed)

    return (r, u)


def extract_run_id(filename: str, model_token: str) -> Optional[str]:
    # filename: <model_token>.<run_id>.json
    if not filename.startswith(model_token + ".") or not filename.endswith(".json"):
        return None
    body = filename[len(model_token) + 1 : -5]
    return body if body else None


def try_get_logs_dir(overrides: Optional[list[str]] = None) -> list[Path]:
    candidates: list[Path] = []
    if overrides:
        for p in overrides:
            if p:
                candidates.append(Path(p))
    try:
        from swebench.harness.constants import RUN_EVALUATION_LOG_DIR  # type: ignore
        candidates.append(Path(RUN_EVALUATION_LOG_DIR))
    except Exception:
        pass
    # Fallbacks
    candidates.append(Path.cwd() / "logs" / "run_evaluation")
    # Common scratch layout on cluster
    scratch = Path("/scratch")
    if scratch.exists():
        candidates.append(scratch / "cbb89" / "logs" / "run_evaluation")
        candidates.append(scratch / "logs" / "run_evaluation")
    return candidates


def count_from_logs(run_id: str, model_token: str, logs_dirs: Optional[list[str]] = None) -> tuple[int, int]:
    # Look in logs dirs for per-instance report.json files
    for base in try_get_logs_dir(logs_dirs):
        run_dir = base / run_id / model_token
        if not run_dir.exists():
            continue
        resolved = 0
        unresolved = 0
        for inst_dir in run_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            report = inst_dir / "report.json"
            if not report.exists():
                continue
            try:
                data = json.loads(report.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict) and len(data) == 1:
                    (iid, payload), = data.items()
                    if isinstance(payload, dict):
                        val = payload.get("resolved")
                        if isinstance(val, bool):
                            if val:
                                resolved += 1
                            else:
                                unresolved += 1
                            continue
                        val = payload.get("Test_Accept")
                        if isinstance(val, bool):
                            if val:
                                resolved += 1
                            else:
                                unresolved += 1
                            continue
            except Exception:
                continue
        return (resolved, unresolved)
    return (0, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize Claude CodeReview results from swebench_eval')
    ap.add_argument('eval_dir', type=str, help='Path to swebench_eval directory')
    ap.add_argument('--out', type=str, default=None, help='Optional output JSON path')
    ap.add_argument('--model-token', type=str, default='openrouter__anthropic__claude-sonnet-4', help='Model token prefix in filenames')
    ap.add_argument('--run-prefix', type=str, default='claude_sonnet_acr_eval_codereview_', help='Run prefix in filenames')
    ap.add_argument('--logs-dir', action='append', default=None, help='Explicit logs/run_evaluation directories to search (can be given multiple times)')
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f'Not found: {eval_dir}')

    files = [p for p in eval_dir.glob('*.json') if is_target_file(p, args.model_token, args.run_prefix)]

    total_files = len(files)
    total_resolved = 0
    total_unresolved = 0

    for p in files:
        r, u = count_from_report(p)
        if (r + u) == 0:
            # Fallback to logs: derive run_id and scan per-instance reports
            run_id = extract_run_id(p.name, args.model_token)
            if run_id:
                r, u = count_from_logs(run_id, args.model_token, args.logs_dir)
        total_resolved += r
        total_unresolved += u

    total = total_resolved + total_unresolved
    rate = (total_resolved / total) if total else 0.0

    print('=== Claude CodeReview Summary ===')
    print(f'Files: {total_files}')
    print(f'Total: {total}, Resolved: {total_resolved}, Unresolved: {total_unresolved}, Resolve@1: {rate:.3f}')

    summary = {
        'mode': 'codereview',
        'model': args.model_token,
        'files': total_files,
        'total': total,
        'resolved': total_resolved,
        'unresolved': total_unresolved,
        'resolve_rate': round(rate, 4),
    }

    out_path = Path(args.out) if args.out else (eval_dir / 'claude_codereview_summary.json')
    out_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'[+] Wrote summary to {out_path}')


if __name__ == '__main__':
    main()


