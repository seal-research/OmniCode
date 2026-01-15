#!/usr/bin/env python3
"""
Hybrid selection with per-repo minimum:
- Ignore instances with any file >30 style errors.
- Ensure all unique symbols are covered (greedy set cover).
- Ensure repo coverage (at least 1 per repo).
- Ensure at least 10 instances per repo if possible.
- Fill remaining quota (default 120) with preference for 10–20 error instances.
- Output symbol coverage mapping, repo coverage mapping, and a .txt file of instance_ids.
"""

import json, random, argparse
from pathlib import Path
from collections import defaultdict


def get_symbols(inst):
    files = inst.get("style_review", {}).get("files", {})
    return {msg["symbol"] for f in files.values() for msg in f.get("messages", [])}


def get_error_count(inst):
    return sum(f.get("message_count", 0) for f in inst.get("style_review", {}).get("files", {}).values())


def is_noisy(inst, max_errors=30):
    files = inst.get("style_review", {}).get("files", {})
    return any(f.get("message_count", 0) > max_errors for f in files.values())


def greedy_symbol_cover(instances):
    all_symbols = set().union(*[get_symbols(i) for i in instances])
    uncovered = set(all_symbols)
    selected = []
    while uncovered:
        best, best_cover = None, set()
        for inst in instances:
            cover = get_symbols(inst) & uncovered
            if len(cover) > len(best_cover):
                best, best_cover = inst, cover
        if not best:
            break
        selected.append(best)
        uncovered -= best_cover
    return selected, all_symbols


def ensure_repo_coverage(selected, instances):
    repos_selected = {s["repo"] for s in selected}
    repos_all = {i["repo"] for i in instances}
    extra = []
    for repo in repos_all - repos_selected:
        candidates = [i for i in instances if i["repo"] == repo]
        ten_twenty = [i for i in candidates if 10 <= get_error_count(i) <= 20]
        choice = random.choice(ten_twenty or candidates)
        extra.append(choice)
    return selected + extra


def build_symbol_mapping(instances):
    mapping = defaultdict(set)
    for inst in instances:
        for sym in get_symbols(inst):
            mapping[sym].add(inst["instance_id"])
    return {sym: sorted(list(ids)) for sym, ids in mapping.items()}


def build_repo_mapping(instances):
    mapping = defaultdict(list)
    for inst in instances:
        mapping[inst["repo"]].append(inst["instance_id"])
    return dict(mapping)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="subset.json")
    ap.add_argument("--symbol-map", default="symbol_mapping.json")
    ap.add_argument("--repo-map", default="repo_mapping.json")
    ap.add_argument("--instances-txt", default="subset_instances.txt", help="Output .txt file of selected instance_ids")
    ap.add_argument("--target-size", type=int, default=120)
    ap.add_argument("--max-errors-per-file", type=int, default=30)
    ap.add_argument("--min-per-repo", type=int, default=10, help="Try to include at least this many per repo if possible")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    clean_data = [i for i in data if not is_noisy(i, args.max_errors_per_file)]
    print(f"Loaded {len(data)} instances, kept {len(clean_data)} after filtering noisy ones.")

    if not clean_data:
        print("No clean instances available, exiting.")
        return

    cover_set, all_symbols = greedy_symbol_cover(clean_data)
    print(f"Symbol coverage selected {len(cover_set)} instances.")

    cover_set = ensure_repo_coverage(cover_set, clean_data)
    print(f"After repo coverage: {len(cover_set)} instances.")

    # Enforce per-repo minimums
    final = list(cover_set)
    repos_needed = defaultdict(int)
    for inst in final:
        repos_needed[inst["repo"]] += 1

    for repo in {i["repo"] for i in clean_data}:
        if repos_needed[repo] < args.min_per_repo:
            candidates = [i for i in clean_data if i["repo"] == repo and i not in final]
            random.shuffle(candidates)
            for inst in candidates:
                if repos_needed[repo] >= args.min_per_repo:
                    break
                final.append(inst)
                repos_needed[repo] += 1

    print(f"After enforcing >= {args.min_per_repo} per repo: {len(final)} instances.")

    # Fill remaining slots
    remaining = [i for i in clean_data if i not in final]
    random.shuffle(remaining)
    bucket_10_20 = [i for i in remaining if 10 <= get_error_count(i) <= 20]
    bucket_other = [i for i in remaining if i not in bucket_10_20]

    for pool in [bucket_10_20, bucket_other]:
        for inst in pool:
            if len(final) >= args.target_size:
                break
            counts = defaultdict(int)
            for f in final:
                counts[f["repo"]] += 1
            if counts[inst["repo"]] < args.target_size * 0.2:
                final.append(inst)
        if len(final) >= args.target_size:
            break

    # Build mappings
    symbol_map = build_symbol_mapping(final)
    repo_map = build_repo_mapping(final)

    # Save outputs
    Path(args.output).write_text(json.dumps(final, indent=2))
    Path(args.symbol_map).write_text(json.dumps(symbol_map, indent=2))
    Path(args.repo_map).write_text(json.dumps(repo_map, indent=2))

    instance_ids = sorted({i["instance_id"] for i in final})
    Path(args.instances_txt).write_text("\n".join(instance_ids) + "\n")

    print(f"Saved {len(final)} instances to {args.output}")
    print(f"Saved symbol mapping to {args.symbol_map}")
    print(f"Saved repo mapping to {args.repo_map}")
    print(f"Saved {len(instance_ids)} unique instance_ids to {args.instances_txt}")

    final_symbols = set().union(*[get_symbols(i) for i in final])
    final_repos = {i["repo"] for i in final}
    print(f"Unique symbols covered: {len(final_symbols)}/{len(all_symbols)}")
    print(f"Repos covered: {len(final_repos)}/{len({i['repo'] for i in clean_data})}")
    uncovered_symbols = all_symbols - final_symbols
    if uncovered_symbols:
        print(f"Uncovered symbols ({len(uncovered_symbols)}): {sorted(uncovered_symbols)}")

    # Print repo histogram
    print("\nInstances per repo:")
    for repo, ids in repo_map.items():
        print(f"  {repo}: {len(ids)}")


if __name__ == "__main__":
    main()
