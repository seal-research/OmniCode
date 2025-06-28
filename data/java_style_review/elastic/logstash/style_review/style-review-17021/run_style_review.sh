#!/bin/bash
set -e

# Function to apply patch and run Checkstyle
run_style_review() {
    local patch_file="$1"
    local output_dir="$2"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Apply the patch
    if [ -f "$patch_file" ]; then
        git apply "$patch_file" || {
            echo "Error applying patch" > "$output_dir/error.log"
            return 1
        }
    else
        echo "No patch file found at $patch_file" > "$output_dir/error.log"
        return 1
    fi
    
    # Find all modified Java files
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
    
    # Create temporary directory for intermediate files
    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT
    
    # Initialize error report array
    echo "[]" > "$output_dir/style_errors.json"
    
    # Initialize counters
    total_errors=0
    total_files=0
    
    # Process each file
    for file in $modified_files; do
        [ -z "$file" ] && continue
        [ ! -f "$file" ] && continue
        
        total_files=$((total_files + 1))
        
        # Run Checkstyle and capture output
        java -jar /usr/local/lib/checkstyle.jar -c /workspace/checkstyle.xml "$file" -f xml > "$temp_dir/checkstyle.xml" 2>/dev/null || true
        
        # Convert XML to JSON for easier processing
        java -cp /usr/local/lib/checkstyle.jar com.puppycrawl.tools.checkstyle.Main -c /workspace/checkstyle.xml "$file" > "$temp_dir/checkstyle.txt" 2>/dev/null || true
        
        # Count errors
        file_errors=$(grep -c "\[ERROR\]" "$temp_dir/checkstyle.txt" || echo 0)
        total_errors=$((total_errors + file_errors))
        
        # Calculate file score (10 - number of errors, minimum 0)
        file_score=$(echo "scale=1; 10 - $file_errors * 0.5" | bc)
        if (( $(echo "$file_score < 0" | bc -l) )); then
            file_score="0.0"
        fi
        
        # Extract error messages
        error_messages=$(grep "\[ERROR\]" "$temp_dir/checkstyle.txt" | sed -e 's/^.*\[ERROR\] //' || echo "")
        
        # Create file report JSON
        file_report="{
            "file": "$file",
            "score": $file_score,
            "error_count": $file_errors,
            "messages": ["
        
        # Add error messages
        first=true
        while IFS= read -r message; do
            [ -z "$message" ] && continue
            
            if $first; then
                first=false
            else
                file_report+=","
            fi
            
            # Extract line number and message
            if [[ "$message" =~ ^([0-9]+):([0-9]+):\ (.*) ]]; then
                line="${BASH_REMATCH[1]}"
                column="${BASH_REMATCH[2]}"
                msg="${BASH_REMATCH[3]}"
                file_report+="{"line": $line, "column": $column, "type": "error", "message": "${msg//"/\"}", "source": "checkstyle"}"
            else
                file_report+="{"line": 0, "column": 0, "type": "error", "message": "${message//"/\"}", "source": "checkstyle"}"
            fi
        done <<< "$error_messages"
        
        file_report+="]},"
        
        # Append to main report (replacing the closing bracket with the new entry and a closing bracket)
        jq -s '.[0] + [.[1]]' "$output_dir/style_errors.json" <(echo "${file_report%?}") > "$temp_dir/new_errors.json"
        mv "$temp_dir/new_errors.json" "$output_dir/style_errors.json"
    done
    
    # Generate final summary
    global_score=10.0
    if [ "$total_files" -gt 0 ]; then
        global_score=$(echo "scale=1; 10 - ($total_errors / $total_files) * 0.5" | bc)
        if (( $(echo "$global_score < 0" | bc -l) )); then
            global_score="0.0"
        fi
    fi
    
    echo "{
        "global_score": $global_score,
        "total_errors": $total_errors,
        "total_warnings": 0
    }" > "$output_dir/style_report.json"
    
    return 0
}

# Main execution
if [ $# -lt 2 ]; then
    echo "Usage: $0 <patch_file> <output_dir>"
    exit 1
fi

run_style_review "$1" "$2"
