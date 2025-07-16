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
RUN apt-get update && apt-get install -y wget unzip git jq

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
    
    def _get_style_review_script(self) -> str:
        return """#!/bin/bash
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
    echo '{"global_score": 10.0,"total_errors": 0,"total_warnings": 0}' > "$safe_output_dir/style_report.json"
    echo "[]" > "$safe_output_dir/style_errors.json"

    # Handle patch application with comprehensive error handling
    if [ -f "$patch_file" ] && [ "$patch_file" != "/dev/null" ]; then
        echo "Applying patch: $patch_file"
        echo "Patch file contents (first 10 lines):"
        head -10 "$patch_file" 2>/dev/null || echo "Could not read patch file"
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

    # Always analyze all Java files in the repo (for both original and patched runs)
    echo "Finding all Java files in the repo to analyze..."
    all_java_files=$(find /workspace/repo -name "*.java" -type f 2>/dev/null)
    total_files=$(echo "$all_java_files" | wc -w)
    total_errors=0
    echo "[]" > "$safe_output_dir/style_errors.json"

    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    for file in $all_java_files; do
        [ -z "$file" ] && continue
        [ ! -f "$file" ] && continue
        file_errors=0
        file_score=10.0
        # Run Checkstyle and capture output
        java -jar /usr/local/lib/checkstyle.jar -c /workspace/checkstyle.xml "$file" -f xml > "$temp_dir/checkstyle.xml" 2>/dev/null || true
        java -cp /usr/local/lib/checkstyle.jar com.puppycrawl.tools.checkstyle.Main -c /workspace/checkstyle.xml "$file" > "$temp_dir/checkstyle.txt" 2>/dev/null || true
        file_errors=$(grep -c "\[ERROR\]" "$temp_dir/checkstyle.txt" || echo 0)
        total_errors=$((total_errors + file_errors))
        file_score=$(echo "scale=1; 10 - $file_errors * 0.5" | bc)
        if (( $(echo "$file_score < 0" | bc -l) )); then
            file_score="0.0"
        fi
        error_messages=$(grep "\[ERROR\]" "$temp_dir/checkstyle.txt" | sed -e 's/^.*\[ERROR\] //' || echo "")
        file_report="{"
        first=true
        while IFS= read -r message; do
            [ -z "$message" ] && continue
            if $first; then
                first=false
            else
                file_report+=",";
            fi
            if [[ "$message" =~ ^([0-9]+):([0-9]+):\ (.*) ]]; then
                line="${BASH_REMATCH[1]}"
                column="${BASH_REMATCH[2]}"
                msg="${BASH_REMATCH[3]}"
                file_report+="\"line\": $line, \"column\": $column, \"type\": \"error\", \"message\": \"${msg//\"/\\\"}\", \"source\": \"checkstyle\"}"
            else
                file_report+="\"line\": 0, \"column\": 0, \"type\": \"error\", \"message\": \"${message//\"/\\\"}\", \"source\": \"checkstyle\"}"
            fi
        done <<< "$error_messages"
        file_report+="}"
        jq -s '.[0] + [.[1]]' "$safe_output_dir/style_errors.json" <(echo "$file_report") > "$temp_dir/new_errors.json"
        mv "$temp_dir/new_errors.json" "$safe_output_dir/style_errors.json"
    done

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
    if [ -n "$output_dir" ]; then
        echo "Copying results to: $output_dir"
        mkdir -p "$output_dir" 2>/dev/null || true
        cp "$safe_output_dir/style_report.json" "$output_dir/original_style_report.json" 2>/dev/null || true
        cp "$safe_output_dir/style_errors.json" "$output_dir/original_style_errors.json" 2>/dev/null || true
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