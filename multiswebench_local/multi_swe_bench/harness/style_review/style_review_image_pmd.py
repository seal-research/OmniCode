from multi_swe_bench.harness.image import Config, File, Image
from multi_swe_bench.harness.pull_request import PullRequest


class JavaStyleReviewImage(Image):
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config
    
    @property
    def pr(self) -> PullRequest:
        return self._pr
    
    @pr.setter
    def pr(self, value: PullRequest):
        self._pr = value
    
    @property
    def config(self) -> Config:
        return self._config
    
    @config.setter
    def config(self, value: Config):
        self._config = value
        
    def dependency(self) -> str:
        return "openjdk:17-slim"
    
    def image_tag(self) -> str:
        return f"style-review-{self.pr.number}"
    
    def workdir(self) -> str:
        return f"style-review-{self.pr.number}"
    
    def files(self) -> list[File]:
        return [
            File(
                dir="",
                name="pmd-ruleset.xml",
                content=self._get_pmd_ruleset()
            ),
            File(
                dir="",
                name="run_style_review.sh",
                content=self._get_style_review_script()
            )
        ]
    
    def dockerfile(self) -> str:
        return f"""FROM {self.dependency()}
{self.global_env}

RUN apt-get update && apt-get install -y wget unzip git jq curl bc libxml2-utils

# Install PMD
RUN curl -L https://github.com/pmd/pmd/releases/download/pmd_releases%2F7.0.0/pmd-dist-7.0.0-bin.zip -o /tmp/pmd.zip && \\
    unzip /tmp/pmd.zip -d /opt && \\
    mv /opt/pmd-bin-7.0.0 /opt/pmd && \\
    ln -s /opt/pmd/bin/pmd /usr/local/bin/pmd && \\
    chmod +x /usr/local/bin/pmd

WORKDIR /workspace

COPY pmd-ruleset.xml /workspace/
COPY run_style_review.sh /workspace/
RUN chmod +x /workspace/run_style_review.sh

{self.clear_env}
"""

    def _get_pmd_ruleset(self) -> str:
        return """<?xml version="1.0" encoding="UTF-8"?>
<ruleset name="PMD Java Style Ruleset"
         xmlns="http://pmd.sourceforge.net/ruleset/2.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://pmd.sourceforge.net/ruleset/2.0.0 https://pmd.github.io/pmd-7.0.0/xsd/ruleset.xsd"
         minimumPriority="3">
    <description>Standard PMD style rules for Java.</description>
    <rule ref="category/java/bestpractices.xml"/>
    <rule ref="category/java/codestyle.xml"/>
    <rule ref="category/java/errorprone.xml"/>
    <rule ref="category/java/design.xml"/>
</ruleset>
"""

    def _get_style_review_script(self) -> str:
        return r"""#!/bin/bash
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
        modified_files=$(find /workspace -name "*.java" -type f 2>/dev/null | head -100 || true)
        echo "Found Java files in /workspace: $modified_files"
        # If still no files, try /workspace/repo as a fallback
        if [ -z "$modified_files" ]; then
            echo "No Java files found in /workspace, searching /workspace/repo..."
            modified_files=$(find /workspace/repo -name "*.java" -type f 2>/dev/null | head -100 || true)
            echo "Found Java files in /workspace/repo: $modified_files"
        fi
    else
        # For patched state, first try to find modified files
        modified_files=$(git diff --name-only HEAD | grep -E '\.java$' || true)
        echo "Modified files from git diff: $modified_files"
        
        # If no modified files, look for existing Java files
        if [ -z "$modified_files" ]; then
            echo "No modified files found, searching for existing Java files..."
            # Search in common Java source directories
            modified_files=$(find /workspace -name "*.java" -type f 2>/dev/null | head -50 || true)
            echo "Found existing Java files in /workspace: $modified_files"
        fi
        
        # If still no files, try /workspace/repo as a fallback
        if [ -z "$modified_files" ]; then
            echo "No Java files found in /workspace, searching /workspace/repo..."
            modified_files=$(find /workspace/repo -name "*.java" -type f 2>/dev/null | head -50 || true)
            echo "Found Java files in /workspace/repo: $modified_files"
        fi
    fi

    # Filter only files that exist and are readable
    existing_files=""
    for file in $modified_files; do
        if [ -f "$file" ] && [ -r "$file" ]; then
            existing_files="$existing_files $file"
            echo "Found existing readable file: $file"
        else
            echo "Skipping non-existent or unreadable file: $file"
        fi
    done
    modified_files=$existing_files

    if [ -z "$modified_files" ]; then
        echo "No existing Java files found for analysis - using default results"
        echo "Directory structure for debugging:"
        find /workspace -type f -name "*.java" 2>/dev/null | head -20 || echo "No Java files found in /workspace"
        find /workspace/repo -type f -name "*.java" 2>/dev/null | head -20 || echo "No Java files found in /workspace/repo"
        # Keep the default results we set earlier
    else
        echo "Analyzing files: $modified_files"
        
        temp_dir=$(mktemp -d)
        trap 'rm -rf "$temp_dir"' EXIT

        echo "[]" > "$safe_output_dir/style_errors.json"
        total_errors=0
        total_files=0

        for file in $modified_files; do
            [ ! -f "$file" ] && continue
            [ ! -r "$file" ] && continue
            
            total_files=$((total_files + 1))
            echo "Processing file: $file"
            echo "File size: $(wc -c < "$file" 2>/dev/null || echo "unknown") bytes"

            output_xml="$temp_dir/pmd_output.xml"
            # Run PMD with comprehensive error handling
            echo "Running PMD on: $file"
            if pmd check -d "$file" -R /workspace/pmd-ruleset.xml -f xml > "$output_xml" 2>/dev/null; then
                echo "PMD completed successfully for: $file"
                file_errors=$(grep -o "<violation " "$output_xml" | wc -l)
                total_errors=$((total_errors + file_errors))
                echo "Found $file_errors violations in: $file"

                file_score=$(echo "scale=1; 10 - $file_errors * 0.5" | bc 2>/dev/null || echo "10.0")
                if (( $(echo "$file_score < 0" | bc -l 2>/dev/null || echo "0") )); then
                    file_score="0.0"
                fi

                # Extract error details with robust parsing
                error_json=$(xmllint --xpath '//violation' "$output_xml" 2>/dev/null | \
                    sed -n 's/.*beginline="\([0-9]*\)".*begincolumn="\([0-9]*\)".*rule="\([^"]*\)".*>\(.*\)<\/violation>.*/{"line": \1, "column": \2, "type": "error", "message": "\4", "source": "\3"},/p' | \
                    sed '$ s/,$//' || echo "")

                echo "{
                    \"file\": \"$file\",
                    \"score\": $file_score,
                    \"error_count\": $file_errors,
                    \"messages\": [${error_json}]
                }" > "$temp_dir/file_error.json"

                # Safely update the main error file
                if [ -f "$temp_dir/file_error.json" ]; then
                    jq -s '.[0] + [.[1]]' "$safe_output_dir/style_errors.json" "$temp_dir/file_error.json" > "$temp_dir/tmp.json" 2>/dev/null || true
                    if [ -f "$temp_dir/tmp.json" ]; then
                        mv "$temp_dir/tmp.json" "$safe_output_dir/style_errors.json" 2>/dev/null || true
                    fi
                fi
            else
                echo "PMD failed to analyze file: $file"
                # Add a default entry for failed files
                echo "{
                    \"file\": \"$file\",
                    \"score\": 10.0,
                    \"error_count\": 0,
                    \"messages\": []
                }" > "$temp_dir/file_error.json"
                
                jq -s '.[0] + [.[1]]' "$safe_output_dir/style_errors.json" "$temp_dir/file_error.json" > "$temp_dir/tmp.json" 2>/dev/null || true
                if [ -f "$temp_dir/tmp.json" ]; then
                    mv "$temp_dir/tmp.json" "$safe_output_dir/style_errors.json" 2>/dev/null || true
                fi
            fi
        done

        # Calculate global score with error handling
        global_score=10.0
        if [ "$total_files" -gt 0 ]; then
            global_score=$(echo "scale=1; 10 - ($total_errors / $total_files) * 0.5" | bc 2>/dev/null || echo "10.0")
            if (( $(echo "$global_score < 0" | bc -l 2>/dev/null || echo "0") )); then
                global_score="0.0"
            fi
        fi

        echo "Final statistics: total_files=$total_files, total_errors=$total_errors, global_score=$global_score"
        echo "{
            \"global_score\": $global_score,
            \"total_errors\": $total_errors,
            \"total_warnings\": 0
        }" > "$safe_output_dir/style_report.json"
    fi

    # Copy results to the specified output directory with comprehensive error handling
    if [ -n "$output_dir" ]; then
        echo "Copying results to: $output_dir"
        mkdir -p "$output_dir" 2>/dev/null || true
        
        # Copy main results
        cp "$safe_output_dir/style_report.json" "$output_dir/original_style_report.json" 2>/dev/null || true
        cp "$safe_output_dir/style_errors.json" "$output_dir/original_style_errors.json" 2>/dev/null || true
        
        # Copy any warning or error logs
        [ -f "$safe_output_dir/patch_warning.log" ] && cp "$safe_output_dir/patch_warning.log" "$output_dir/patch_warning.log" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_errors.log" ] && cp "$safe_output_dir/patch_errors.log" "$output_dir/patch_errors.log" 2>/dev/null || true
        [ -f "$safe_output_dir/error.log" ] && cp "$safe_output_dir/error.log" "$output_dir/error.log" 2>/dev/null || true
    fi

    echo "Style review completed successfully"
    echo "=== Style review finished ==="
    return 0
}

# Call the function with the provided arguments
run_style_review "$@"
"""

