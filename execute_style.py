import subprocess
import os

# Configurable inputs
'''pull_numbers = [4641, 4615, 4487, 4486, 4469, 4468, 4426, 4365, 4360, 4338, 4325, 4320, 4311, 4304, 4257, 4230, 4228, 
4219, 4189, 4186, 4159, 4132, 4131, 4087, 4072, 4050, 4048, 4015, 4013, 3860, 3716, 3701, 3666, 3626, 3625, 3621, 3560,
3509, 3371, 2036, 1923, 3851]'''
#pull_numbers = [ 43330, 42856, 42852, 42174, 42067, 41213]
#pull_numbers = [3424, 3220, 3173, 3167, 3133, 3129]
#pull_numbers = [10638, 11781, 7041]
#pull_numbers = [2775, 2559, 2285, 2097, 1245, 82]
#pull_numbers = [6586, 2945]
#pull_numbers = [1309, 1263, 1208, 1204, 1182, 1172, 1142, 1053, 1016, 964, 922, 891, 729, 566, 370, 183, 174, 980]
#pull_numbers = [644, 638, 590, 544, 531]
pull_numbers = [4144, 4035, 2542, 2536, 2688]
org_repo = "googlecontainertools/jib"

# Command to run the style review
codearena_cmd_template = (
    "python codearena.py --StyleReview "
    "--predictions_path gold "
    "--run_id mswe_java_style_review "
    "--max_workers 1 "
    '--instance_ids "{org_repo}:{pull}" '
    "--mswe_phase all "
    "--force_rebuild True "
    "--review_type pmd"
)

# Path template to locate the style review logs
log_parser_path = "log_parser.py"
style_log_path_template = "data/java_style_review/{repo_path}/style_review/style-review-{pull}"

# Run codearena and log parser
for pull in pull_numbers:
    instance = f"{org_repo}:{pull}"
    org, repo = org_repo.split("/")
    repo_path = f"{org}/{repo}"

    print(f"\n🔧 Running style review for PR: {instance}")

    for i in range(1):
        print(f"  → Run {i + 1} with codearena...")
        cmd = codearena_cmd_template.format(org_repo=org_repo, pull=pull)
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Run {i + 1} for PR {pull} failed with error code {e.returncode}")

    # Run log parser on result directory
    review_dir = style_log_path_template.format(repo_path=repo_path, pull=pull)
    print(f"  🧾 Parsing style logs for {review_dir}...")

    try:
        subprocess.run(f"python {log_parser_path} {review_dir}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Log parsing for PR {pull} failed with error code {e.returncode}")

print("\n✅ All PRs processed.")

