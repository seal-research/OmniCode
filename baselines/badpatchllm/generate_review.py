# USAGE
# python baselines/badpatchllm/generate_rev.py --input_tasks data/codearena_instances.json --output_dir baselines/badpatchllm/logs/gemini_outputs --instance_ids fastapi__fastapi-1549 --model_name gemini-2.0-flash --api_key
import os
import json
import argparse
from pathlib import Path
import google.generativeai as genai

def query_llm_for_review(patch: str, problem_statement: str, correct_patch_example: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Queries the LLM to get a detailed review of the provided patch using Google's Generative AI.
    The prompt now includes the problem statement and a correct patch example so the LLM knows the intended changes.
    
    Args:
        patch (str): The bad patch text.
        problem_statement (str): A description of the problem that needs to be fixed.
        correct_patch_example (str): An example of a correct patch.
        model_name (str): The model name to use (default: "gemini-2.0-flash").
    
    Returns:
        str: The generated detailed review.
    """
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are an experienced software engineer tasked with reviewing code patches. "
        "Below is a problem statement, a correct patch example, and a submitted patch which is likely incorrect or incomplete. "
        "Please provide a detailed review that includes:\n"
        "  1. A brief summary of the problem to be fixed.\n"
        "  2. Identification of issues with the submitted patch (e.g., missing context, incorrect modifications, or potential bugs).\n"
        "  3. Specific suggestions for improvements that would bring the patch closer to the correct solution. You can create these suggestions via a comparison of the submitted incorrect patch against the correct patch example, but do NOT justify with reasoning along the line of: the correct patch does this.\n\n"
        "Problem Statement:\n"
        f"{problem_statement}\n\n"
        "Correct Patch Example:\n"
        f"{correct_patch_example}\n\n"
        "Submitted Patch (Bad Patch):\n"
        f"{patch}\n\n"
        "Detailed Review:"
    )
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.9, "top_p": 0.9} 
    )
    
    return response.text.strip()

def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    instance_ids: list,
    secret_key: str,
    num_reviews: int
):
    # Configure the Google Generative AI library with the provided API key.
    genai.configure(api_key=secret_key)

    # Load the input tasks JSON to build a mapping from instance_id to its context.
    try:
        with open(input_tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        # Assume tasks is a list of objects, each having an "instance_id", "problem_statement", and either "correct_patch_example" or "patch" key.
        instance_map = {}
        for task in tasks:
            inst_id = str(task.get("instance_id"))
            if inst_id:
                instance_map[inst_id] = {
                    "problem_statement": task.get("problem_statement", ""),
                    # Use the key "correct_patch_example" if available; otherwise, fallback to "patch" field.
                    "correct_patch_example": task.get("correct_patch_example", task.get("patch", ""))
                }
    except Exception as e:
        raise ValueError(f"Failed to load input tasks from {input_tasks_path}: {e}")
    output_dir_path = output_dir_path / instance_ids[0]
    # If instance_ids was not provided on the command line, use all instance_ids from the JSON.
    if not instance_ids:
        instance_ids = list(instance_map.keys())
    patch_file = output_dir_path / f"patch_1.diff"
    # Process each instance.
    for instance_id in instance_ids:
        # Read the bad patch from a file named patch_<instance_id>.diff in the output directory.
        try:
            with open(patch_file, "r", encoding="utf-8") as f:
                bad_patch = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Bad patch file for instance {instance_id} not found at {patch_file}. Skipping.")
            continue

        if not bad_patch:
            print(f"Warning: Patch file for instance {instance_id} is empty. Skipping.")
            continue

        # Retrieve additional context from the input tasks.
        context = instance_map.get(instance_id, {})
        problem_statement = context.get("problem_statement", "No problem statement provided.")
        correct_patch_example = context.get("correct_patch_example", "No correct patch example provided.")

        print(f"Processing instance: {instance_id}")
        print("Loaded Bad Patch:")
        print(bad_patch)

        # Generate the requested number of reviews for this patch.
        reviews = []
        for i in range(num_reviews):
            review = query_llm_for_review(
                patch=bad_patch,
                problem_statement=problem_statement,
                correct_patch_example=correct_patch_example,
                model_name=model_name
            )
            reviews.append(review)
            print(f"\nLLM Review {i+1} for instance {instance_id}:")
            print(review)

        # Save the reviews for this instance to a file.
        review_file = output_dir_path / f"{instance_id}_reviews.txt"
        try:
            with open(review_file, "w", encoding="utf-8") as f:
                for idx, review in enumerate(reviews, start=1):
                    f.write(f"Review {idx}:\n{review}\n\n")
            print(f"Reviews for instance {instance_id} saved to {review_file}\n")
        except Exception as e:
            print(f"Error saving reviews for instance {instance_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detailed reviews for bad patches using Gemini-2.0-Flash.")
    parser.add_argument("--input_tasks", required=True, help="Path to the JSON file containing input tasks.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory where bad patch files are stored and reviews will be saved.")
    parser.add_argument("--model_name", default="gemini-2.0-flash", help="Model name to use for review generation.")
    parser.add_argument("--instance_ids", default=None, help="Comma-separated list of instance IDs to process. If omitted, all instance IDs from the input tasks file will be used.")
    parser.add_argument("--api_key", required=True, help="Secret key for accessing the Google Generative AI API.")
    parser.add_argument("--num_reviews", type=int, default=1, help="Number of reviews to generate per patch.")
    
    args = parser.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else [],
        secret_key=args.api_key,
        num_reviews=args.num_reviews,
    )