#!/usr/bin/env python3
import subprocess, os, shutil

SCRIPT = "python_repo_pylint_runner.py"
OUTDIR = "pylint_results"
os.makedirs(OUTDIR, exist_ok=True)

repos = {
    "scrapy__scrapy":      ("https://github.com/scrapy/scrapy.git", 6542),
    "sphinx-doc__sphinx":  ("https://github.com/sphinx-doc/sphinx.git", 11510),
    "sympy__sympy":        ("https://github.com/sympy/sympy.git", 24661),
    "ytdl-org__youtube-dl":("https://github.com/ytdl-org/youtube-dl.git", 32987),
}

for key, (url, pr) in repos.items():
    out_path = os.path.join(OUTDIR, f"{key}-{pr}.json")
    workdir = '.'
    print(f"[+] Running pylint review for {key} (PR {pr}) in {workdir}")
    repo_name = key.split('_')[-1]

    ### Phase 1: collecting and running pylint on dataset
    try:
        subprocess.run([
            "python", SCRIPT,
            "--repo-url", url,
            "--pr", str(pr),
            "--out", out_path,
            "--work-dir", workdir,
            "--instance-id", f'{key}-{pr}'
        ], check=False)
    finally:
        # cleanup: remove workdir + cloned repo
        print(f"[+] Completed pylint review for {key}_{pr}")
        shutil.rmtree(f'./repo', ignore_errors=True)

    ### Phase 2: (optional) re-filtering dataset (The dataset is filtered by default during Phase 1)
    # try:
    #     subprocess.run([
    #         "python", 'filter_pylint_results.py',
    #         "--instance-id", f'{key}-{pr}',
    #         "--input", f'pylint_results/full_pylint_run_data/full_{key.split('__')[-1]}_pylint.json'
    #     ], check=False)
    # finally:
    #     # cleanup: remove workdir + cloned repo
    #     print(f"[+] Completed pylint review for {key}_{pr}")
    #     shutil.rmtree(f'./repo', ignore_errors=True)

    ### Phase 3: splitting dataset
    try:
        subprocess.run([
            "python", 'python_style_review_dataset_generator.py',
            "--instance-id", f'{key}-{pr}',
            "--results", f'pylint_results/{key}-{pr}.json',
            "--output", f'python_style_review_dataset/{key}-{pr}_review_instances.json', 
            # "--batch-size", '20' # default = 0 => one file per instance
        ], check=False)
    finally:
        # cleanup: remove workdir + cloned repo
        print(f"[+] Completed pylint review for {key}_{pr}")
        shutil.rmtree(f'./repo', ignore_errors=True)

