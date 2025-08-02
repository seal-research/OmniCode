import json
import os

def filter_and_append_jsonl(
    error_report_path,
    overview_path,
    output_jsonl_path,
    rejected_sources,
    label
):
    # Load detailed error report
    with open(error_report_path, 'r') as f:
        data = json.load(f)

    # Load original overview (not used now, since we recompute)
    with open(overview_path, 'r') as f:
        _ = json.load(f)

    filtered_files = []
    total_errors = 0
    total_warnings = 0

    for file_entry in data:
        # 🔒 Skip non-Java files
        if file_entry["file"] is not None and (not file_entry["file"].endswith(".java")):
            continue

        # Filter messages
        filtered_messages = [
            msg for msg in file_entry["messages"]
            if msg["source"] not in rejected_sources
        ]

        error_count = sum(1 for msg in filtered_messages if msg["type"] == "error")
        warning_count = sum(1 for msg in filtered_messages if msg["type"] == "warning")

        # Only include file if some messages remain
        if filtered_messages:
            file_score = max(0, 10 - (error_count / 2))

            filtered_files.append({
                "file": file_entry["file"],
                "score": round(file_score, 2),
                "error_count": error_count,
                "messages": filtered_messages
            })

            total_errors += error_count
            total_warnings += warning_count

    num_files = len(filtered_files)
    global_score = max(0, 10 - (total_errors / num_files * 0.5)) if num_files > 0 else 0.0

    # Overview after all filters
    new_overview = {
        "global_score": round(global_score, 2),
        "total_errors": total_errors,
        "total_warnings": total_warnings
    }

    # Build output entry
    output_entry = {
        "label": label,
        "overview": new_overview,
        "files": filtered_files
    }

    # Append to output JSONL
    with open(output_jsonl_path, 'a') as out_f:
        out_f.write(json.dumps(output_entry) + '\n')

    print(f"[✓] Appended filtered report under label: {label}")
    print(f"    - Included Java files: {num_files}")
    print(f"    - Total errors after filter: {total_errors}")
    print(f"    - Global score: {round(global_score, 2)}")


# === USAGE ===
if __name__ == "__main__":
    pr=input('Enter pr')
    org="googlecontainertools"
    repo="jib"
    #directory='/mnt/e/'+org+"/"+repo+"/style_review/"+"style-review-"+pr+"/"
    directory='data/java_style_review/'+org+"/"+repo+"/style_review/"+"style-review-"+pr+"/"
    error_json_path = directory+"original_style_errors.json"
    overview_json_path = directory+"original_style_report.json"
    output_jsonl_file = "original_results_pmd.jsonl"
    pr_label = org+"/"+repo+":pr-"+pr
    #rejected = ["com.puppycrawl.tools.checkstyle.checks.NewlineAtEndOfFileCheck", "com.puppycrawl.tools.checkstyle.checks.whitespace.FileTabCharacterCheck","com.puppycrawl.tools.checkstyle.checks.NewlineAtEndOfFileCheck",
    #"com.puppycrawl.tools.checkstyle.checks.whitespace.WhitespaceAroundCheck","com.puppycrawl.tools.checkstyle.checks.whitespace.WhitespaceAfterCheck","com.puppycrawl.tools.checkstyle.checks.coding.MagicNumberCheck"]
    rejected = ["LongVariable","TooManyMethods","ShortVariable"]
    filter_and_append_jsonl(
        error_report_path=error_json_path,
        overview_path=overview_json_path,
        output_jsonl_path=output_jsonl_file,
        rejected_sources=rejected,
        label=pr_label
    )

