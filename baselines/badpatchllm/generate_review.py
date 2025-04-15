import os
import json
import argparse
from pathlib import Path
import google.generativeai as genai
import math
import copy # Import copy for deep copying

def query_llm_for_review(bad_patch: str, problem_statement: str, correct_patch_example: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Queries the LLM to get a detailed review of the provided patch using Google's Generative AI.
    The prompt now includes the problem statement and a correct patch example so the LLM knows the intended changes.

    Args:
        bad_patch (str): The bad patch text.
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
        "Please provide a detailed review that identify of issues with the submitted patch (e.g., missing context, incorrect modifications, or potential bugs) and specify suggestions for improvements that would bring the patch closer to the correct solution. You can create these suggestions via a comparison of the submitted incorrect patch against the correct patch example, but do NOT justify with reasoning along the line of: the correct patch does this. Some example reviews you should output include: `object passed to as_scalar_or_list_str can be a single element list. I this case, should extract element and display instead of printing list` or `Using incorrect shape for cright. It should be of shape (noutp, right.shape[1])` or `arguments list passed convert to world values should be unrolled instead of a list` which you can see are concise and to the point, you can be a bit more in-depth but be sure to not give away the actual answer (code from correct patch example should not be shared).\n\n"
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
        # print(f"Error: JSON file not found at {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                # print(f"Error: Expected a JSON list, but found {type(data)} in {json_path}")
                return None
            return data
    except json.JSONDecodeError:
        # print(f"Error: Could not decode JSON from {json_path}")
        return None
    except Exception as e:
        # print(f"An unexpected error occurred while loading {json_path}: {e}")
        return None

def main(
    json_file_path: Path, # Renamed for clarity
    model_name: str,
    secret_key: str,
    num_reviews_per_patch: int # Kept for consistency, but logic will only use the first review
):
    # Configure the Google Generative AI library with the provided API key.
    genai.configure(api_key=secret_key)

    # Load the instances from the JSON file.
    instances = load_instances(json_file_path)
    if not instances:
        # print("No instances loaded. Exiting.")
        return

    modified_instances = copy.deepcopy(instances) # Work on a copy

    # Process each instance.
    for instance in modified_instances: # Limit to first 3 instances for testing
        instance_id = instance.get("instance_id")
        problem_statement = instance.get("problem_statement", "No problem statement provided.")
        correct_patch_example = instance.get("patch", "No correct patch example provided.")
        bad_patches_list = instance.get("bad_patch")
        reviews = instance.get("Review", [])

        if not instance_id:
            continue

        # --- Validate the bad_patch field ---
        if bad_patches_list is None or (isinstance(bad_patches_list, float) and math.isnan(bad_patches_list)):
            continue

        # --- Check if bad patches already have reviews, if so don't overwrite/do redundantwork ---
        if reviews and len(reviews) >= num_reviews_per_patch* len(bad_patches_list):
            print(f"  Info: Instance {instance_id} already has enough reviews. Skipping review generation.")
            continue


        if not isinstance(bad_patches_list, list):
            if isinstance(bad_patches_list, str):
                 bad_patches_list = [bad_patches_list]
            else:
                # print(f"Warning: Instance {instance_id} 'bad_patch' field is not a list or string (type: {type(bad_patches_list)}). Skipping review generation.")
                continue

        if not bad_patches_list:
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
        # instance["Review_Author"] = "Gemini 2.0 Flash" # Hardcoded author
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

    # print("-" * 70)
    # print(f"Finished processing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reviews for bad patches listed in a JSON file and update the file.")
    # Changed argument name and help text
    parser.add_argument("--input_tasks", required=True, help="Path to the codearena_instances.json file (will be read and updated).")
    # Removed output_dir argument
    parser.add_argument("--model_name", default="gemini-2.0-flash", help="Model name to use for review generation (e.g., gemini-2.0-flash).")
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