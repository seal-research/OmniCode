#!/usr/bin/env python3
"""Sample bad patches and reviews from OmniCode benchmark for human annotation."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "annotation"

LANGUAGE_FILES = {
    "python": "omnicode_instances_python.json",
    "java": "omnicode_instances_java.json",
    "cpp": "omnicode_instances_cpp.json",
}

EMPTY_ANNOTATIONS = {
    "patch_incorrectness_type": None,
    "patch_plausibility": None,
    "review_info_leakage": None,
    "review_correctness": None,
    "notes": "",
}


def load_instances(language: str) -> list[dict]:
    path = DATA_DIR / LANGUAGE_FILES[language]
    with open(path) as f:
        return json.load(f)


def sample_for_language(instances: list[dict], language: str, n: int, rng: random.Random) -> list[dict]:
    with_bad_patches = [inst for inst in instances if inst.get("bad_patches")]
    print(f"  {language}: {len(with_bad_patches)} instances with bad patches (out of {len(instances)} total)")

    if len(with_bad_patches) < n:
        print(f"  WARNING: only {len(with_bad_patches)} available, sampling all")
        sampled = with_bad_patches
    else:
        sampled = rng.sample(with_bad_patches, n)

    results = []
    for inst in sampled:
        bad_patch = rng.choice(inst["bad_patches"])
        results.append({
            "language": language,
            "repo": inst["repo"],
            "instance_id": inst["instance_id"],
            "problem_statement": inst["problem_statement"],
            "gold_patch": inst["patch"],
            "bad_patch": {
                "idx": bad_patch["idx"],
                "source": bad_patch["source"],
                "patch": bad_patch["patch"],
            },
            "review": bad_patch["review"],
            "annotations": dict(EMPTY_ANNOTATIONS),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Sample bad patches for annotation")
    parser.add_argument("--n", type=int, default=10, help="Number of instances per language")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "sampled_instances.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_instances = []
    print(f"Sampling {args.n} instances per language (seed={args.seed}):")
    for language in LANGUAGE_FILES:
        instances = load_instances(language)
        sampled = sample_for_language(instances, language, args.n, rng)
        all_instances.extend(sampled)

    output = {
        "metadata": {
            "sampled_at": datetime.now().isoformat(),
            "n_per_language": args.n,
            "seed": args.seed,
            "total_instances": len(all_instances),
        },
        "instances": all_instances,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(all_instances)} instances to {output_path}")


if __name__ == "__main__":
    main()
