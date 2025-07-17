import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
import concurrent.futures
from typing import Dict, Optional

from tqdm import tqdm

from multi_swe_bench.harness.pull_request import PullRequest, Base
from multi_swe_bench.harness.image import Config
from multi_swe_bench.utils import docker_util
from multi_swe_bench.utils.logger import setup_logger
from swebench.harness.utils import str2bool

from style_review_instance import JavaStyleReviewInstance
from style_review_report import JavaStyleReviewReport, StyleReviewSummary, StyleFileReport, StyleIssue

import xml.etree.ElementTree as ET
import re

def load_predictions(predictions_path: str) -> Dict[str, dict]:
    """Load predictions from a file."""
    predictions = {}
    
    if predictions_path == 'gold':
        # Gold predictions are handled elsewhere
        return predictions
        
    with open(predictions_path, 'r') as f:
        if predictions_path.endswith('.json'):
            preds = json.load(f)
            for pred in preds:
                instance_id = f"{pred.get('org')}/{pred.get('repo')}:{pred.get('number')}"
                predictions[instance_id] = pred
        elif predictions_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    pred = json.loads(line)
                    instance_id = f"{pred.get('org')}/{pred.get('repo')}:{pred.get('number')}"
                    predictions[instance_id] = pred
    
    return predictions

def load_dataset(dataset_path: str, instance_ids: Optional[list] = None) -> Dict[str, dict]:
    """Load dataset from a file."""
    dataset = {}
    
    with open(dataset_path, 'r') as f:
        if dataset_path.endswith('.json'):
            data = json.load(f)
            for item in data:
                instance_id = f"{item.get('org')}/{item.get('repo')}:{item.get('number')}"
                if not instance_ids or instance_id in instance_ids:
                    dataset[instance_id] = item
        elif dataset_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    instance_id = f"{item.get('org')}/{item.get('repo')}:{item.get('number')}"
                    if not instance_ids or instance_id in instance_ids:
                        dataset[instance_id] = item
    
    return dataset

def create_default_style_report_json():
    """Create a default style report JSON"""
    return json.dumps({
        "global_score": 10.0,
        "total_errors": 0,
        "total_warnings": 0
    })

def create_default_style_errors_json():
    """Create a default style errors JSON"""
    return json.dumps([])

def extract_checkstyle_xml_from_log(log_path):
    """Extract Checkstyle XML output from the log file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start, end = None, None
    for i, line in enumerate(lines):
        if '==== FULL CHECKSTYLE VIOLATION XML OUTPUT ====' in line:
            start = i + 1
        if '==== END OF CHECKSTYLE VIOLATION XML OUTPUT ====' in line:
            end = i
            break
    if start is not None and end is not None and start < end:
        xml_str = ''.join(lines[start:end]).strip()
        return xml_str
    return None

def parse_checkstyle_xml_string_to_json(xml_str):
    """Parse Checkstyle XML string to the expected JSON format."""
    if not xml_str:
        return []
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return []
    files = []
    for file_elem in root.findall('file'):
        file_path = file_elem.get('name', '')
        messages = []
        for err in file_elem.findall('error'):
            messages.append({
                "line": int(err.get("line", 0)),
                "column": int(err.get("column", 0)),
                "type": "error",
                "message": err.get("message", ""),
                "source": err.get("source", "checkstyle")
            })
        files.append({
            "file": file_path,
            "score": max(0.0, 10 - 0.5 * len(messages)),
            "error_count": len(messages),
            "messages": messages
        })
    return files

def extract_checkstyle_stats_from_log(log_path):
    """Extract total_files, total_errors, and global_score from the log file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Final statistics:'):
                # Example: Final statistics: total_files=271, total_errors=5649, global_score=0.0
                m = re.search(r'total_files=(\d+), total_errors=(\d+), global_score=([0-9.]+)', line)
                if m:
                    return {
                        "global_score": float(m.group(3)),
                        "total_errors": int(m.group(2)),
                        "total_warnings": 0
                    }
    return None

def run_style_review(
    instance: JavaStyleReviewInstance, 
    workdir: Path, 
    log_dir: Path, 
    run_id: str,
    timeout: int,
    force_rebuild: bool
) -> Optional[JavaStyleReviewReport]:
    """Run style review for a single instance."""
    # Create a safe file name by replacing special characters
    safe_id = instance.pr.id.replace(':', '-').replace('/', '_')
    
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger with a safe path
    logger = setup_logger(log_dir, f"{safe_id}_style_review.log", "INFO", True)
    
    # Create a safe instance directory path
    instance_dir = workdir / instance.pr.org / instance.pr.repo / "style_review" / instance.dependency().workdir()
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up paths for patches and outputs
    fix_patch_path = instance_dir.absolute() / "fix.patch"
    original_report_path = instance_dir / "original_style_report.json"
    original_errors_path = instance_dir / "original_style_errors.json"
    patched_report_path = instance_dir / "patched_style_report.json"
    patched_errors_path = instance_dir / "patched_style_errors.json"
    
    with open(fix_patch_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(instance.pr.fix_patch)
    
    # Check if Docker image exists, build if not or if force_rebuild is True
    image_name = instance.dependency().image_full_name()
    if force_rebuild or not docker_util.exists(image_name):
        logger.info(f"Building image {image_name}...")
        # Build Dockerfile
        dockerfile_path = instance_dir / "Dockerfile"
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(instance.dependency().dockerfile())
        
        # Copy files needed for the image
        for file in instance.dependency().files():
            file_path = instance_dir / file.dir / file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file.content)
        
        # Build the image
        try:
            docker_util.build(instance_dir, "Dockerfile", image_name, logger)
        except Exception as e:
            logger.error(f"Error building image {image_name}: {e}")
            return None
    
    # Clone the repo if not already present
    repo_dir = instance_dir / "repo"
    if not repo_dir.exists():
        clone_url = f"https://github.com/{instance.pr.org}/{instance.pr.repo}.git"
        try:
            subprocess.run(["git", "clone", clone_url, str(repo_dir)], check=True)
            subprocess.run(["git", "checkout", instance.pr.base.sha], cwd=repo_dir, check=True)
        except Exception as e:
            logger.error(f"Error cloning or checking out repo: {e}")
            return None
    else:
        # Optionally, check if the repo is at the correct commit, and reset if not
        try:
            current_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir
            ).decode().strip()
            if current_sha != instance.pr.base.sha:
                subprocess.run(["git", "fetch"], cwd=repo_dir, check=True)
                subprocess.run(["git", "checkout", instance.pr.base.sha], cwd=repo_dir, check=True)
        except Exception as e:
            logger.error(f"Error ensuring repo is at correct commit: {e}")
            return None
    
    # Run style review
    logger.info(f"Running style review for {instance.pr.id}...")
    
    try:
        # Always extract and overwrite error JSONs from logs for Checkstyle
        orig_log_path = instance_dir / "original_run.log"
        patched_log_path = instance_dir / "patched_run.log"
        orig_xml = extract_checkstyle_xml_from_log(orig_log_path)
        patched_xml = extract_checkstyle_xml_from_log(patched_log_path)
        orig_json = parse_checkstyle_xml_string_to_json(orig_xml)
        patched_json = parse_checkstyle_xml_string_to_json(patched_xml)
        with open(original_errors_path, "w", encoding="utf-8") as f:
            json.dump(orig_json, f, indent=2)
        with open(patched_errors_path, "w", encoding="utf-8") as f:
            json.dump(patched_json, f, indent=2)

        # Always extract and overwrite report JSONs from logs for Checkstyle
        orig_stats = extract_checkstyle_stats_from_log(orig_log_path)
        patched_stats = extract_checkstyle_stats_from_log(patched_log_path)
        if orig_stats:
            with open(original_report_path, "w", encoding="utf-8") as f:
                json.dump(orig_stats, f, indent=2)
        if patched_stats:
            with open(patched_report_path, "w", encoding="utf-8") as f:
                json.dump(patched_stats, f, indent=2)
        
        # Run original style check
        logger.info("Running initial style check (without patch)...")
        try:
            original_output = docker_util.run(
                image_name,
                instance.run(),
                instance_dir / "original_run.log",
                volumes=[
                    f"{str(fix_patch_path.absolute())}:{instance.dependency().fix_patch_path()}:rw",
                    f"{str(repo_dir.absolute())}:/workspace/repo:rw",
                    f"{str((instance_dir / 'output').absolute())}:/workspace/output:rw"
                ]
            )
            logger.info("Original style check completed successfully")
        except Exception as e:
            logger.error(f"Error running original style check: {e}")
        
        # Run patched style check
        logger.info("Running style check with patch applied...")
        try:
            patched_output = docker_util.run(
                image_name,
                instance.fix_patch_run(),
                instance_dir / "patched_run.log",
                volumes=[
                    f"{str(fix_patch_path.absolute())}:{instance.dependency().fix_patch_path()}:rw",
                    f"{str(repo_dir.absolute())}:/workspace/repo:rw",
                    f"{str((instance_dir / 'output').absolute())}:/workspace/output:rw"
                ]
            )
            logger.info("Patched style check completed successfully")
        except Exception as e:
            logger.error(f"Error running patched style check: {e}")
                
        logger.info("Processing style review results...")
        
        # Load and parse reports
        try:
            # After writing original_report_path and patched_report_path, always update style_review_report.json
            try:
                with open(original_report_path, "r") as f:
                    original_summary = StyleReviewSummary(**json.load(f))
                with open(patched_report_path, "r") as f:
                    patched_summary = StyleReviewSummary(**json.load(f))
                with open(original_errors_path, "r") as f:
                    original_issues_data = json.load(f)
                    original_issues = []
                    for issue_data in original_issues_data:
                        file_issues = []
                        for msg in issue_data.get("messages", []):
                            file_issues.append(StyleIssue(
                                line=msg.get("line", 0),
                                column=msg.get("column", 0),
                                type=msg.get("type", "error"),
                                message=msg.get("message", ""),
                                source=msg.get("source", "checkstyle")
                            ))
                        original_issues.append(StyleFileReport(
                            file=issue_data.get("file", ""),
                            score=issue_data.get("score", 0.0),
                            error_count=issue_data.get("error_count", 0),
                            messages=file_issues
                        ))
                with open(patched_errors_path, "r") as f:
                    patched_issues_data = json.load(f)
                    patched_issues = []
                    for issue_data in patched_issues_data:
                        file_issues = []
                        for msg in issue_data.get("messages", []):
                            file_issues.append(StyleIssue(
                                line=msg.get("line", 0),
                                column=msg.get("column", 0),
                                type=msg.get("type", "error"),
                                message=msg.get("message", ""),
                                source=msg.get("source", "checkstyle")
                            ))
                        patched_issues.append(StyleFileReport(
                            file=issue_data.get("file", ""),
                            score=issue_data.get("score", 0.0),
                            error_count=issue_data.get("error_count", 0),
                            messages=file_issues
                        ))
                report = JavaStyleReviewReport(
                    org=instance.pr.org,
                    repo=instance.pr.repo,
                    number=instance.pr.number,
                    original_score=original_summary,
                    patched_score=patched_summary,
                    original_issues=original_issues,
                    patched_issues=patched_issues
                )
                report.calculate_improvement()
                with open(instance_dir / "style_review_report.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(report.__dict__, indent=2))
            except Exception as e:
                logger.error(f"Error updating style_review_report.json: {e}")
            
            # Save report
            with open(instance_dir / "style_review_report.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(report.__dict__, indent=2))
            
            logger.info(f"Style review completed for {instance.pr.id}")
            logger.info(f"Original score: {original_summary.global_score}, Patched score: {patched_summary.global_score}")
            logger.info(f"Improvement: {report.improvement}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error processing style review results: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error running style review: {e}")
        return None

def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    min_score: float,
    max_severity: str
):
    """Main function to run the style review."""
    # Set up directories
    workdir = Path("./data/java_style_review")
    workdir.mkdir(parents=True, exist_ok=True)
    
    log_dir = workdir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(log_dir, "java_style_review.log", "INFO", True)
    logger.info(f"Starting Java StyleReview with run_id: {run_id}")
    
    # Load dataset and predictions
    logger.info(f"Loading dataset from {dataset_name}")
    dataset = load_dataset(dataset_name, instance_ids)
    
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = load_predictions(predictions_path)
    
    # Create instances
    instances = []
    config = Config(need_clone=True, global_env=None, clear_env=True)
    
    for instance_id, data in dataset.items():
        if instance_id in predictions:
            try:
                # Handle the base field properly
                base_data = data.get("base", {})
                if isinstance(base_data, dict):
                    # Create proper Base object
                    base = Base(
                        label=base_data.get("label", ""),
                        ref=base_data.get("ref", ""),
                        sha=base_data.get("sha", "")
                    )
                else:
                    # Create default Base object
                    base = Base(label="", ref="", sha="")
                
                # Handle resolved_issues field
                resolved_issues = data.get("resolved_issues", [])
                if not isinstance(resolved_issues, list):
                    resolved_issues = []
                
                # Create PullRequest object
                pr = PullRequest(
                    org=str(data.get("org", "")),
                    repo=str(data.get("repo", "")),
                    number=int(data.get("number", 0)),
                    state=data.get("state", ""),
                    title=data.get("title", ""),
                    body=data.get("body", ""),
                    base=base,  # Use the properly created Base object
                    resolved_issues=resolved_issues,
                    fix_patch=predictions[instance_id].get("patch", ""),
                    test_patch=""  # No test patch needed for style review
                )
                
                # Create StyleReviewInstance
                instance = JavaStyleReviewInstance(pr, config)
                instances.append(instance)
                logger.info(f"Successfully created instance for {instance_id}")
            except Exception as e:
                logger.error(f"Error creating instance for {instance_id}: {e}")
    
    logger.info(f"Created {len(instances)} instances for style review")
    
    # Run style review in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(
                run_style_review, 
                instance, 
                workdir, 
                log_dir, 
                run_id,
                timeout,
                force_rebuild
            ): instance
            for instance in instances
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_instance), total=len(instances)):
            instance = future_to_instance[future]
            try:
                report = future.result()
                if report:
                    results.append(report)
            except Exception as e:
                logger.error(f"Error running style review for {instance.pr.id}: {e}")
    
    # Generate final report
    resolved_instances = []
    unresolved_instances = []
    
    for report in results:
        if min_score is not None and report.patched_score.global_score < min_score:
            unresolved_instances.append(report.id)
        elif report.improvement and report.improvement > 0:
            resolved_instances.append(report.id)
        else:
            unresolved_instances.append(report.id)
    
    final_report = {
        "total_instances": len(instances),
        "completed_instances": len(results),
        "resolved_instances": len(resolved_instances),
        "unresolved_instances": len(unresolved_instances),
        "average_improvement": sum(r.improvement or 0 for r in results) / len(results) if results else 0,
        "resolved_ids": sorted(resolved_instances),
        "unresolved_ids": sorted(unresolved_instances),
        "criteria": {
            "min_score": min_score,
            "max_severity": max_severity
        }
    }
    
    # Fix the report name to avoid duplicate 'java_style_review'
    final_report_path = log_dir / f"{run_id}_report.json"
    with open(final_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    
    logger.info(f"Java StyleReview completed for {len(results)} instances")
    logger.info(f"Resolved: {len(resolved_instances)}, Unresolved: {len(unresolved_instances)}")
    logger.info(f"Final report written to {final_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Java StyleReview")
    parser.add_argument("--dataset_name", default="data/codearena_instances.json", help="Name of the dataset")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers")
    parser.add_argument("--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images")
    parser.add_argument("--cache_level", type=str, choices=["none", "base", "env", "instance"], help="Cache level", default="env")
    parser.add_argument("--clean", type=str2bool, default=False, help="Clean images above cache level")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout for running tests for each instance")
    parser.add_argument("--min_score", type=float, default=None, help="Minimum acceptable style score (0-10)")
    parser.add_argument("--max_severity", type=str, choices=['convention', 'warning', 'error'], default=None, help="Maximum acceptable severity level")
    
    args = parser.parse_args()
    
    main(**vars(args))