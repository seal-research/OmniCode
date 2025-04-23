"""
How to run this script:
1. Make sure you have the required libraries installed:
   pip install google-generativeai
2. Set up your Google Generative AI API key in the environment variable or pass it directly.
3. Run the script with the following command:
python [PATH to /codearena/baselines/badpatchllm/generate_review.py] \
    --input_tasks [PATH to codearena_instances.json or desired file] \
    --api_key [API_KEY] 

Optionally, you can specify the model name and the number of reviews per patch:
    --model_name [MODEL_NAME] \
    --num_reviews_per_patch [NUM_REVIEWS]
"""
import os
import json
import argparse
from pathlib import Path
import google.generativeai as genai
import math
import copy # Import copy for deep copying

def query_llm_for_review(bad_patch: str, problem_statement: str, correct_patch_example: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> str:
    """
    Queries the LLM to get a detailed review of the provided patch using Google's Generative AI.
    The prompt now includes the problem statement and a correct patch example so the LLM knows the intended changes.

    Args:
        bad_patch (str): The bad patch text.
        problem_statement (str): A description of the problem that needs to be fixed.
        correct_patch_example (str): An example of a correct patch.
        model_name (str): The model name to use (default: "gemini-2.5-flash-preview-04-17").

    Returns:
        str: The generated detailed review.
    """
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are an experienced software engineer tasked with reviewing code patches. "
        "Below is a problem statement, a correct patch example, and a submitted patch which is likely incorrect or incomplete. "
        "Please provide a detailed review that identify of issues with the submitted patch (e.g., missing context, incorrect modifications, or potential bugs) and specify suggestions for improvements the new patch"
        "would solve the problem statement. You can create these suggestions via a comparison of the submitted incorrect patch against the correct"
        "patch example, but try to come up with original suggestions on your own (as there can be multiple ways to fix th problem. DO NOT justify with reasoning along the line of: the correct patch does this. Some example reviews you"
        "should output include: \"object passed to as_scalar_or_list_str can be a single element list. I this case, should extract "
        "element and display instead of printing list\" or \"Using incorrect shape for cright. It should be of shape (noutp, right.shape[1])\""
        "or \"arguments list passed convert to world values should be unrolled instead of a list\" which you can see are concise and to the point,"
        "you should keep your recommendation/response under 50 words and make sure to not give away the actual answer as code from correct patch example should not be shared.\n\n"
        "Problem Statement:\n"
        f"{problem_statement}\n\n"
        "Correct Patch Example:\n"
        f"{correct_patch_example}\n\n"
        "Submitted Patch (Bad Patch):\n"
        f"{bad_patch}\n\n"
        "Detailed Review:"
    )

    review_text = "[Review not generated]" # Default value
    api_response = None # Initialize api_response to None
    try:
        api_response = model.generate_content(
            prompt,
            # Adjust generation config as needed, ensure compatibility with the chosen model
            # Example generation config (adjust temperature, top_p, etc. as needed)
            generation_config={"temperature": 0.7, "top_p": 0.9}
        )
        # Accessing response.text might raise an exception if the response is blocked
        # or doesn't contain valid text. Use response.parts to check first.
        if api_response.parts:
             review_text = api_response.text.strip()
        else:
            # Handle cases where the response might be blocked or empty
            # Check for prompt feedback if available
            if hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback:
                review_text = f"[Review generation failed due to safety settings or other issues: {api_response.prompt_feedback}]"
            else:
                review_text = "[Review generation failed: Empty response]"

    except Exception as e:
        # Handle exceptions during the API call or response processing
        review_text = f"[Error generating review: {e}]"
        # Optionally log more details about the error or the response if it exists
        # (but be careful as 'api_response' might still be None if the exception
        # happened before or during the generate_content call)
        # if api_response and hasattr(api_response, 'prompt_feedback') and api_response.prompt_feedback:
        #     pass # Log api_response.prompt_feedback if needed
        # else:
        #     pass # Log basic error 'e'

    return review_text

def load_instances(json_path: Path) -> list:
    """Loads instance data from the specified JSON file."""
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Expected a JSON list, but found {type(data)} in {json_path}")
                return None
            return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {json_path}: {e}")
        return None

def validate_and_normalize_list(value, instance_id, field_name):
    """
    Validates and normalizes the input value to always be a list if valid.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None  # Explicitly no data provided

    if isinstance(value, str):
        # Convert string to list, handle empty string as empty list
        return [value] if value else []

    if isinstance(value, list):
        return value

    # If none of the above, it's an invalid type
    print(f"Warning: Instance {instance_id} '{field_name}' field is not None, str, or list (type: {type(value)}). Skipping.")
    return None

def main(
    json_file_path: Path,
    model_name: str,
    secret_key: str,
    num_reviews_per_patch: int
):
    genai.configure(api_key=secret_key)
    # ... (setup code like genai.configure, load_instances) ...
    instances = load_instances(json_file_path)
    if not instances:
        return

    modified_instances = copy.deepcopy(instances) # Work on a copy

    for instance in modified_instances[:3]: # Example: process first 3
        instance_id = instance.get("instance_id")
        problem_statement = instance.get("problem_statement", "No problem statement provided.")
        correct_patch_example = instance.get("patch", "No correct patch example provided.")
        bad_patches_input = instance.get("bad_patch")
        reviews_input = instance.get("Review")

        if not instance_id:
            print("Warning: Skipping instance with no instance_id.")
            continue

        # Validate and normalize bad_patches
        bad_patches_list = validate_and_normalize_list(bad_patches_input, instance_id, 'bad_patch')

        # If bad_patches_list is None (invalid input) or empty, skip review generation
        if bad_patches_list is None or not bad_patches_list:
            print(f"  Info: No valid bad patches found for instance {instance_id}. Skipping review generation.")
            continue

        # Validate and normalize existing reviews
        reviews_list = validate_and_normalize_list(reviews_input, instance_id, 'Review')

        # If reviews_list is None, it means the field was invalid or missing. Treat as 0 reviews.
        # If it's an empty list [], it means 0 reviews exist.
        existing_review_count = len(reviews_list) if reviews_list is not None else 0

        # Check if enough reviews already exist
        # Note: This assumes num_reviews_per_patch applies *per bad patch*.
        # If it means total reviews for the instance, the logic might differ slightly.
        required_reviews = num_reviews_per_patch * len(bad_patches_list)
        if existing_review_count >= required_reviews:
            print(f"  Info: Instance {instance_id} already has {existing_review_count} reviews (>= {required_reviews} required). Skipping.")
            continue

        new_reviews = []
        for bad_patch_content in bad_patches_list:
            if isinstance(bad_patch_content, str) and bad_patch_content.strip():
                review = query_llm_for_review(
                    bad_patch=bad_patch_content,
                    problem_statement=problem_statement,
                    correct_patch_example=correct_patch_example,
                    model_name=model_name
                )
                new_reviews.append(review)
        # Update the instance dictionary
        instance["Review"] = new_reviews
        # instance["Review_Author"] = "Gemini 2.5 Flash" # Hardcoded author
            # else:
                # print(f"  Info: No valid bad patches found for instance {instance_id}. Skipping review generation.")


    # Write the modified data back to the original JSON file.
    try:
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(modified_instances, f, indent=4) # Use indent for readability
        # print(f"\nSuccessfully updated {json_file_path} with generated reviews.")
    except Exception as e:
        # print(f"Error writing updated data back to {json_file_path}: {e}")
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reviews for bad patches listed in a JSON file and update the file.")
    # Changed argument name and help text
    parser.add_argument("--input_tasks", required=True, help="Path to the codearena_instances.json file (will be read and updated).")
    # Removed output_dir argument
    parser.add_argument("--model_name", default="gemini-2.5-flash-preview-04-17", help="Model name to use for review generation (e.g., gemini-2.5-flash).")
    parser.add_argument("--api_key", required=True, help="Secret key for accessing the Google Generative AI API.")
    # Kept this argument but clarified its behavior in the help text
    parser.add_argument("--num_reviews_per_patch", type=int, default=1, help="Number of reviews to attempt generating per bad patch (currently only the first successful review for the first valid patch per instance is saved).")

    args = parser.parse_args()

    main(
        json_file_path=Path(args.input_tasks),
        model_name=args.model_name,
        secret_key=args.api_key,
        num_reviews_per_patch=args.num_reviews_per_patch,
    )