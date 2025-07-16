#!/bin/bash
set -e

run_style_review() {
    local patch_file="$1"
    local output_dir="$2"

    echo "=== Starting style review ==="
    echo "Patch file: $patch_file"
    echo "Output directory: $output_dir"
    echo "Current working directory: $(pwd)"
    echo "Workspace contents (ls -la /workspace):"
    ls -la /workspace/ 2>/dev/null || echo "No /workspace directory"
    echo "Full directory tree (find /workspace):"
    find /workspace -type d 2>/dev/null || echo "No directories found"
    echo "All files in /workspace (ls -lR /workspace):"
    ls -lR /workspace 2>/dev/null || echo "No files found"

    # Make a safe directory derived from output_dir — never use raw input path directly
    safe_output_dir="/workspace/output_dir_$(date +%s%N)"
    echo "Safe output directory: $safe_output_dir"

    echo "Using mkdir"
    mkdir -p "$safe_output_dir"

    # Initialize default results immediately
    echo '{
        "global_score": 10.0,
        "total_errors": 0,
        "total_warnings": 0
    }' > "$safe_output_dir/style_report.json"
    echo "[]" > "$safe_output_dir/style_errors.json"

    # Handle patch application with comprehensive error handling
    if [ -f "$patch_file" ] && [ "$patch_file" != "/dev/null" ]; then
        echo "Applying patch: $patch_file"
        echo "Patch file contents (first 10 lines):"
        head -10 "$patch_file" 2>/dev/null || echo "Could not read patch file"
        
        # Apply patch and capture any errors, but never fail the script
        patch_errors_file="$safe_output_dir/patch_errors.log"
        if ! git apply --reject --whitespace=fix "$patch_file" 2>"$patch_errors_file" 2>&1; then
            echo "Warning: Patch could not be fully applied. Some files may be missing or already patched." > "$safe_output_dir/patch_warning.log"
            echo "Patch application errors:" >> "$safe_output_dir/patch_warning.log"
            if [ -f "$patch_errors_file" ]; then
                cat "$patch_errors_file" >> "$safe_output_dir/patch_warning.log" 2>/dev/null || true
            fi
            echo "Continuing with analysis despite patch issues..."
        else
            echo "Patch applied successfully"
        fi
    elif [ "$patch_file" = "/dev/null" ]; then
        echo "No patch to apply (original state)"
    else
        echo "No patch file found at $patch_file" > "$safe_output_dir/error.log"
        echo "Continuing with analysis without patch..."
    fi

    # Find Java files to analyze - try multiple approaches
    echo "Finding Java files to analyze..."
    # For original state (no patch), analyze all Java files in the repository
    if [ "$patch_file" = "/dev/null" ]; then
        echo "Original state analysis - looking for all Java files in repository..."
        # Search for all Java files in the repository
        java_dirs=$(find /workspace/repo -type d -name java 2>/dev/null | head -10 || true)
        if [ -z "$java_dirs" ]; then
            java_dirs="/workspace/repo"
        fi
    else
        # For patched state, analyze all Java files as well
        java_dirs=$(find /workspace/repo -type d -name java 2>/dev/null | head -10 || true)
        if [ -z "$java_dirs" ]; then
            java_dirs="/workspace/repo"
        fi
    fi

    # Find all Java files to analyze (for accurate total_files count)
    all_java_files=$(find $java_dirs -name "*.java" -type f 2>/dev/null)
    total_files=$(echo "$all_java_files" | wc -w)
    total_errors=0
    echo "[]" > "$safe_output_dir/style_errors.json"

    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT
    pmd_error_log="$safe_output_dir/pmd_error.log"
    pmd_output_xml="$temp_dir/pmd_output.xml"
    echo "Running PMD on: $java_dirs"
    # Use -r option to suppress progress bar warning and output to file
    if ! pmd check -d $java_dirs -R /workspace/pmd-ruleset.xml -f xml -r "$pmd_output_xml" 2> "$pmd_error_log"; then
        echo "PMD failed to analyze some files. See $pmd_error_log for details."
    fi

    # Count total violations directly from the XML for robust scoring
    if [ -s "$pmd_output_xml" ]; then
        total_errors=$(grep -c '<violation ' "$pmd_output_xml")
    fi

    # Parse PMD XML output and build style_errors.json (per-file breakdown)
    if [ -s "$pmd_output_xml" ]; then
        # Use xmllint to extract all <file> nodes and their violations
        xmllint --xpath '//file' "$pmd_output_xml" 2>/dev/null | \
        awk -v q="\"" 'BEGIN{RS="<file ";FS="</file>"} NR>1{print "<file " $1}' | while read -r file_block; do
            file_path=$(echo "$file_block" | grep -o 'name="[^"]*"' | head -1 | cut -d'"' -f2)
            error_count=$(echo "$file_block" | grep -c '<violation ')
            file_score=$(echo "scale=1; 10 - $error_count * 0.5" | bc 2>/dev/null || echo "10.0")
            if (( $(echo "$file_score < 0" | bc -l 2>/dev/null || echo "0") )); then
                file_score="0.0"
            fi
            # Extract all violation details for this file
            error_json="["
            while read -r vline; do
                # Extract attributes and message
                beginline=$(echo "$vline" | grep -o 'beginline="[^"]*"' | cut -d'"' -f2)
                begincolumn=$(echo "$vline" | grep -o 'begincolumn="[^"]*"' | cut -d'"' -f2)
                rule=$(echo "$vline" | grep -o 'rule="[^"]*"' | cut -d'"' -f2)
                message=$(echo "$vline" | sed -n 's/.*<violation[^>]*>\(.*\)<\/violation>.*/\1/p' | sed 's/"/\\"/g')
                if [ -n "$error_json" ] && [ "$error_json" != "[" ]; then
                    error_json+=",";
                fi
                error_json+="{\"line\": ${beginline:-0}, \"column\": ${begincolumn:-0}, \"type\": \"error\", \"message\": \"${message}\", \"source\": \"${rule}\"}"
            done < <(echo "$file_block" | grep '<violation ')
            error_json+="]"
            # Write file report JSON
            file_report="{\n  \"file\": \"$file_path\", \"score\": $file_score, \"error_count\": $error_count, \"messages\": $error_json\n}"
            jq -s '.[0] + [.[1]]' "$safe_output_dir/style_errors.json" <(echo "$file_report") > "$temp_dir/tmp.json" 2>/dev/null || true
            if [ -f "$temp_dir/tmp.json" ]; then
                mv "$temp_dir/tmp.json" "$safe_output_dir/style_errors.json" 2>/dev/null || true
            fi
        done
    fi

    # Calculate global score with error handling
    global_score=10.0
    if [ "$total_files" -gt 0 ]; then
        global_score=$(echo "scale=1; 10 - ($total_errors / $total_files) * 0.5" | bc 2>/dev/null || echo "10.0")
        if (( $(echo "$global_score < 0" | bc -l 2>/dev/null || echo "0") )); then
            global_score="0.0"
        fi
    fi

    echo "Final statistics: total_files=$total_files, total_errors=$total_errors, global_score=$global_score"
    echo "{\n    \"global_score\": $global_score,\n    \"total_errors\": $total_errors,\n    \"total_warnings\": 0\n}" > "$safe_output_dir/style_report.json"

    # Copy results to the specified output directory with comprehensive error handling
    if [ -n "$output_dir" ]; then
        echo "Copying results to: $output_dir"
        mkdir -p "$output_dir" 2>/dev/null || true
        cp "$safe_output_dir/style_report.json" "$output_dir/original_style_report.json" 2>/dev/null || true
        cp "$safe_output_dir/style_errors.json" "$output_dir/original_style_errors.json" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_warning.log" ] && cp "$safe_output_dir/patch_warning.log" "$output_dir/patch_warning.log" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_errors.log" ] && cp "$safe_output_dir/patch_errors.log" "$output_dir/patch_errors.log" 2>/dev/null || true
        [ -f "$safe_output_dir/error.log" ] && cp "$safe_output_dir/error.log" "$output_dir/error.log" 2>/dev/null || true
        [ -f "$pmd_error_log" ] && cp "$pmd_error_log" "$output_dir/pmd_error.log" 2>/dev/null || true
        # Copy the full PMD XML output
        [ -f "$pmd_output_xml" ] && cp "$pmd_output_xml" "$output_dir/pmd_output.xml" 2>/dev/null || true
    fi

    echo "\n==== FULL PMD VIOLATION XML OUTPUT ===="
    if [ -f "$pmd_output_xml" ]; then
        cat "$pmd_output_xml"
    else
        echo "No PMD XML output found."
    fi
    echo "==== END OF PMD VIOLATION XML OUTPUT ===="

    echo "Style review completed successfully"
    echo "=== Style review finished ==="
    return 0
}

# Call the function with the provided arguments
run_style_review "$@"
