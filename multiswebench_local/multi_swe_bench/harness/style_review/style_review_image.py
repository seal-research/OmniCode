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
        # Return base Java image
        return "openjdk:17-slim"
    
    def image_tag(self) -> str:
        return f"style-review-{self.pr.number}"
    
    def workdir(self) -> str:
        return f"style-review-{self.pr.number}"
    
    def files(self) -> list[File]:
        # Include Checkstyle configuration and any utility scripts
        return [
            File(
                dir="",
                name="checkstyle.xml",
                content=self._get_checkstyle_config()
            ),
            File(
                dir="",
                name="run_style_review.sh",
                content=self._get_style_review_script()
            )
        ]
    
    def dockerfile(self) -> str:
        # Create a Dockerfile that installs Checkstyle and sets up environment
        return f"""FROM {self.dependency()}
{self.global_env}

# Install necessary tools
RUN apt-get update && apt-get install -y wget unzip git jq bc

# Install Checkstyle
RUN wget -q https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.12.1/checkstyle-10.12.1-all.jar -O /usr/local/lib/checkstyle.jar

# Set up working directory
WORKDIR /workspace

# Copy configuration files
COPY checkstyle.xml /workspace/
COPY run_style_review.sh /workspace/
RUN chmod +x /workspace/run_style_review.sh

{self.clear_env}
"""
    
    def _get_checkstyle_config(self) -> str:
        # Return a standard Java checkstyle config
        return """<?xml version="1.0"?>
<!DOCTYPE module PUBLIC "-//Checkstyle//DTD Checkstyle Configuration 1.3//EN" "https://checkstyle.org/dtds/configuration_1_3.dtd">
<module name="Checker">
    <property name="severity" value="error"/>
    <module name="TreeWalker">
        <!-- Basic style checks -->
        <module name="ConstantName"/>
        <module name="LocalVariableName"/>
        <module name="MemberName"/>
        <module name="MethodName"/>
        <module name="PackageName"/>
        <module name="ParameterName"/>
        <module name="StaticVariableName"/>
        <module name="TypeName"/>
        
        <!-- Code quality checks -->
        <module name="AvoidStarImport"/>
        <module name="IllegalImport"/>
        <module name="RedundantImport"/>
        <module name="UnusedImports"/>
        <module name="MethodLength"/>
        <module name="ParameterNumber"/>
        <module name="EmptyBlock"/>
        <module name="NeedBraces"/>
        <module name="LeftCurly"/>
        <module name="RightCurly"/>
        <module name="WhitespaceAround"/>
        <module name="WhitespaceAfter"/>
        <module name="NoWhitespaceAfter"/>
        <module name="NoWhitespaceBefore"/>
        <module name="OperatorWrap"/>
        <module name="ParenPad"/>
        <module name="TypecastParenPad"/>
        <module name="ModifierOrder"/>
        <module name="RedundantModifier"/>
        <module name="AvoidNestedBlocks"/>
        <module name="EmptyStatement"/>
        <module name="EqualsHashCode"/>
        <module name="HiddenField"/>
        <module name="IllegalInstantiation"/>
        <module name="MagicNumber"/>
        <module name="MissingSwitchDefault"/>
        <module name="SimplifyBooleanExpression"/>
        <module name="SimplifyBooleanReturn"/>
        <module name="FinalClass"/>
        <module name="HideUtilityClassConstructor"/>
        <module name="InterfaceIsType"/>
        <module name="VisibilityModifier"/>
        <module name="ArrayTypeStyle"/>
        <module name="TodoComment"/>
        <module name="UpperEll"/>
    </module>
    
    <!-- File-level checks -->
    <module name="FileTabCharacter"/>
    <module name="NewlineAtEndOfFile"/>
</module>
"""
    
    def fix_patch_path(self) -> str:
        return "/home/fix.patch"
    
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

    # Make a safe directory for all intermediate output
    safe_output_dir="/workspace/output_dir_$(date +%s%N)"
    echo "Safe output directory: $safe_output_dir"
    mkdir -p "$safe_output_dir"

    # Initialize default results immediately
    echo '{"global_score": 10.0,"total_errors": 0,"total_warnings": 0}' > "$safe_output_dir/style_report.json"
    echo "[]" > "$safe_output_dir/style_errors.json"

    # Handle patch application with comprehensive error handling
    PATCH_STATUS="not_attempted"
    if [ -f "$patch_file" ] && [ "$patch_file" != "/dev/null" ]; then
        echo "[DEBUG] Directory tree under /workspace before patch:"
        find /workspace | sort
        echo "[DEBUG] Directory tree under /workspace/repo before patch:"
        find /workspace/repo | sort
        echo "[DEBUG] Listing /workspace/repo before patch:"
        ls -l /workspace/repo
        echo "[DEBUG] Current working directory: $(pwd)"
        echo "[DEBUG] Patch file contents (first 20 lines):"
        head -20 "$patch_file" 2>/dev/null || echo "Could not read patch file"
        # Print first 20 lines of each file the patch will touch (if exists)
        echo "[DEBUG] Attempting to print first 20 lines of each file to be patched:"
        grep '^+++ ' "$patch_file" | awk '{print $2}' | sed 's|^b/||' | while read -r f; do
            if [ -f "/workspace/repo/$f" ]; then
                echo "[DEBUG] File: /workspace/repo/$f (first 20 lines):"
                head -20 "/workspace/repo/$f"
            else
                echo "[DEBUG] File: /workspace/repo/$f does not exist."
            fi
        done
        echo "Applying patch: $patch_file"
        echo "Patch file contents (first 10 lines):"
        head -10 "$patch_file" 2>/dev/null || echo "Could not read patch file"
        patch_errors_file="$safe_output_dir/patch_errors.log"
        if (cd /workspace/repo && git apply --check "$patch_file" 2>"$patch_errors_file"); then
            if (cd /workspace/repo && git apply --reject --whitespace=fix "$patch_file" 2>>"$patch_errors_file"); then
                echo "Patch applied successfully" | tee -a "$safe_output_dir/patch_status.log"
                PATCH_STATUS="applied"
            else
                echo "Patch partially applied or with warnings. See $patch_errors_file for details." | tee -a "$safe_output_dir/patch_status.log"
                PATCH_STATUS="partial"
                echo "--- PATCH ERRORS BEGIN ---"
                cat "$patch_errors_file"
                echo "--- PATCH ERRORS END ---"
            fi
        else
            echo "Patch could NOT be applied at all. See $patch_errors_file for details." | tee -a "$safe_output_dir/patch_status.log"
            PATCH_STATUS="failed"
            echo "--- PATCH ERRORS BEGIN ---"
            cat "$patch_errors_file"
            echo "--- PATCH ERRORS END ---"
        fi
        echo "PATCH_STATUS=$PATCH_STATUS" | tee -a "$safe_output_dir/patch_status.log"
    elif [ "$patch_file" = "/dev/null" ]; then
        echo "No patch to apply (original state)" | tee -a "$safe_output_dir/patch_status.log"
        PATCH_STATUS="none"
        echo "PATCH_STATUS=$PATCH_STATUS" | tee -a "$safe_output_dir/patch_status.log"
    else
        echo "No patch file found at $patch_file" > "$safe_output_dir/error.log"
        echo "Continuing with analysis without patch..." | tee -a "$safe_output_dir/patch_status.log"
        PATCH_STATUS="missing"
        echo "PATCH_STATUS=$PATCH_STATUS" | tee -a "$safe_output_dir/patch_status.log"
    fi

    # Always analyze all Java files in the repo (for both original and patched runs)
    echo "Finding all Java files in the repo to analyze..."
    all_java_files=$(find /workspace/repo -name "*.java" -type f 2>/dev/null)
    total_files=$(echo "$all_java_files" | wc -w)
    total_errors=0
    echo "[]" > "$safe_output_dir/style_errors.json"

    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT
    checkstyle_xml="$temp_dir/checkstyle_output.xml"
    checkstyle_error_log="$safe_output_dir/checkstyle_error.log"

    # Run Checkstyle on the entire repo at once, output XML
    echo "Running Checkstyle on /workspace/repo ..."
    if ! java -jar /usr/local/lib/checkstyle.jar -c /workspace/checkstyle.xml /workspace/repo -f xml > "$checkstyle_xml" 2> "$checkstyle_error_log"; then
        echo "Checkstyle failed to analyze some files. See $checkstyle_error_log for details."
    fi

    # Parse Checkstyle XML output and build style_errors.json (per-file breakdown)
    if [ -s "$checkstyle_xml" ]; then
        # Use xmllint to extract all <file> nodes and their errors
        xmllint --xpath '//file' "$checkstyle_xml" 2>/dev/null | \
        awk -v q="\"" 'BEGIN{RS="<file ";FS="</file>"} NR>1{print "<file " $1}' | while read -r file_block; do
            file_path=$(echo "$file_block" | grep -o 'name="[^"]*"' | head -1 | cut -d'"' -f2)
            error_count=$(echo "$file_block" | grep -c '<error ')
            file_score=$(echo "scale=1; 10 - $error_count * 0.5" | bc 2>/dev/null || echo "10.0")
            if (( $(echo "$file_score < 0" | bc -l 2>/dev/null || echo "0") )); then
                file_score="0.0"
            fi
            # Extract all error details for this file
            error_json="["
            while read -r eline; do
                # Extract attributes and message
                line=$(echo "$eline" | grep -o 'line="[^"]*"' | cut -d'"' -f2)
                column=$(echo "$eline" | grep -o 'column="[^"]*"' | cut -d'"' -f2)
                message=$(echo "$eline" | grep -o 'message="[^"]*"' | sed 's/message=\"\([^\"]*\)\"/\1/' | sed 's/"/\\"/g')
                source=$(echo "$eline" | grep -o 'source="[^"]*"' | cut -d'"' -f2)
                if [ -n "$error_json" ] && [ "$error_json" != "[" ]; then
                    error_json+=",";
                fi
                error_json+="{\"line\": ${line:-0}, \"column\": ${column:-0}, \"type\": \"error\", \"message\": \"${message}\", \"source\": \"${source}\"}"
            done < <(echo "$file_block" | grep '<error ')
            error_json+="]"
            # Write file report JSON
            file_report="{\n  \"file\": \"$file_path\", \"score\": $file_score, \"error_count\": $error_count, \"messages\": $error_json\n}"
            jq -s '.[0] + [.[1]]' "$safe_output_dir/style_errors.json" <(echo "$file_report") > "$temp_dir/tmp.json" 2>/dev/null || true
            if [ -f "$temp_dir/tmp.json" ]; then
                mv "$temp_dir/tmp.json" "$safe_output_dir/style_errors.json" 2>/dev/null || true
            fi
        done
    fi

    # Count total errors directly from the XML for robust scoring
    if [ -s "$checkstyle_xml" ]; then
        total_errors=$(grep -c '<error ' "$checkstyle_xml")
    fi

    # Generate final summary
    global_score=10.0
    if [ "$total_files" -gt 0 ]; then
        global_score=$(echo "scale=1; 10 - ($total_errors / $total_files) * 0.5" | bc)
        if (( $(echo "$global_score < 0" | bc -l) )); then
            global_score="0.0"
        fi
    fi

    echo "Final statistics: total_files=$total_files, total_errors=$total_errors, global_score=$global_score"
    echo "{"
    echo "    \"global_score\": $global_score,"
    echo "    \"total_errors\": $total_errors,"
    echo "    \"total_warnings\": 0"
    echo "}" > "$safe_output_dir/style_report.json"

    # Copy results to the specified output directory with comprehensive error handling
    if [ -n "$output_dir" ] && [ "$output_dir" != "/dev/null" ]; then
        echo "Copying results to: $output_dir"
        mkdir -p "$output_dir" 2>/dev/null || true
        cp "$safe_output_dir/style_report.json" "$output_dir/original_style_report.json" 2>/dev/null || true
        cp "$safe_output_dir/style_errors.json" "$output_dir/original_style_errors.json" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_warning.log" ] && cp "$safe_output_dir/patch_warning.log" "$output_dir/patch_warning.log" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_errors.log" ] && cp "$safe_output_dir/patch_errors.log" "$output_dir/patch_errors.log" 2>/dev/null || true
        [ -f "$safe_output_dir/error.log" ] && cp "$safe_output_dir/error.log" "$output_dir/error.log" 2>/dev/null || true
        [ -f "$checkstyle_error_log" ] && cp "$checkstyle_error_log" "$output_dir/checkstyle_error.log" 2>/dev/null || true
        [ -f "$checkstyle_xml" ] && cp "$checkstyle_xml" "$output_dir/checkstyle_output.xml" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_status.log" ] && cp "$safe_output_dir/patch_status.log" "$output_dir/patch_status.log" 2>/dev/null || true
    fi

    echo "\n==== FULL CHECKSTYLE VIOLATION XML OUTPUT ===="
    if [ -f "$checkstyle_xml" ]; then
        cat "$checkstyle_xml"
    else
        echo "No Checkstyle XML output found."
    fi
    echo "==== END OF CHECKSTYLE VIOLATION XML OUTPUT ===="

    echo "Style review completed successfully"
    echo "=== Style review finished ==="
    return 0
}

# Call the function with the provided arguments
run_style_review "$@"
"""