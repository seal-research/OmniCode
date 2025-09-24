#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    resolved = data.get('resolved_ids')
    unresolved = data.get('unresolved_ids')
    completed = data.get('completed_ids')

    r = len(resolved) if isinstance(resolved, list) else 0
    u = len(unresolved) if isinstance(unresolved, list) else 0

    if r == 0 and u == 0 and isinstance(completed, list):
        # If lists are missing, conservatively count all completed as unresolved
        u = len(completed)

    return (r, u)


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize Claude CodeReview results from swebench_eval')
    ap.add_argument('eval_dir', type=str, help='Path to swebench_eval directory')
    ap.add_argument('--out', type=str, default=None, help='Optional output JSON path')
    ap.add_argument('--model-token', type=str, default='openrouter__anthropic__claude-sonnet-4', help='Model token prefix in filenames')
    ap.add_argument('--run-prefix', type=str, default='claude_sonnet_acr_eval_codereview_', help='Run prefix in filenames')
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


