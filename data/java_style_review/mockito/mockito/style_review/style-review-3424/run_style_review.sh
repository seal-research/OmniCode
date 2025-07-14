#!/bin/bash
set -e

run_style_review() {
    local patch_file="$1"
    local output_dir="$2"

    mkdir -p "$output_dir"

    if [ -f "$patch_file" ]; then
        git apply "$patch_file" || {
            echo "Error applying patch" > "$output_dir/error.log"
            return 1
        }
    else
        echo "No patch file found at $patch_file" > "$output_dir/error.log"
        return 1
    fi

    modified_files=$(git diff --name-only HEAD | grep -E '\.java$' || true)

    if [ -z "$modified_files" ]; then
        echo '{
            "global_score": 10.0,
            "total_errors": 0,
            "total_warnings": 0
        }' > "$output_dir/style_report.json"
        echo "[]" > "$output_dir/style_errors.json"
        return 0
    fi

    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    echo "[]" > "$output_dir/style_errors.json"
    total_errors=0
    total_files=0

    for file in $modified_files; do
        [ ! -f "$file" ] && continue
        total_files=$((total_files + 1))

        output_xml="$temp_dir/pmd_output.xml"
        pmd check -d "$file" -R /workspace/pmd-ruleset.xml -f xml > "$output_xml" || true

        file_errors=$(grep -o "<violation " "$output_xml" | wc -l)
        total_errors=$((total_errors + file_errors))

        file_score=$(echo "scale=1; 10 - $file_errors * 0.5" | bc)
        if (( $(echo "$file_score < 0" | bc -l) )); then
            file_score="0.0"
        fi

        error_json=$(xmllint --xpath '//violation' "$output_xml" 2>/dev/null | \
            sed -n 's/.*beginline="\([0-9]*\)".*begincolumn="\([0-9]*\)".*rule="\([^"]*\)".*>\(.*\)<\/violation>.*/{"line": \1, "column": \2, "type": "error", "message": "\4", "source": "\3"},/p' | \
            sed '$ s/,$//')

        echo "{
            \"file\": \"$file\",
            \"score\": $file_score,
            \"error_count\": $file_errors,
            \"messages\": [${error_json}]
        }" > "$temp_dir/file_error.json"

        jq -s '.[0] + [.[1]]' "$output_dir/style_errors.json" "$temp_dir/file_error.json" > "$temp_dir/tmp.json"
        mv "$temp_dir/tmp.json" "$output_dir/style_errors.json"
    done

    global_score=10.0
    if [ "$total_files" -gt 0 ]; then
        global_score=$(echo "scale=1; 10 - ($total_errors / $total_files) * 0.5" | bc)
        if (( $(echo "$global_score < 0" | bc -l) )); then
            global_score="0.0"
        fi
    fi

    echo "{
        \"global_score\": $global_score,
        \"total_errors\": $total_errors,
        \"total_warnings\": 0
    }" > "$output_dir/style_report.json"

    return 0
}

if [ $# -lt 2 ]; then
    echo "Usage: $0 <patch_file> <output_dir>"
    exit 1
fi

run_style_review "$1" "$2"
