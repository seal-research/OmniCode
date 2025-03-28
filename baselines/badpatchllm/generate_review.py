#!/usr/bin/env python3
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig # For specifying config

def process_input_data():
    """
    Process or generate the input data required for patch generation.
    In a real scenario, this function would gather information from a pull request,
    issue, or another source. For this example, we simulate input data with a dictionary.
    """
    input_data = {
        "problem_statement": (
            "I need a way to specify servers in the OpenAPI spec. "
            "I want to be able to use the generated openapi.json doc as it is and hook it up with a document publishing flow, "
            "but I'm not able to because I have to add in information about servers manually."
        ),
        "desired_patch_description": ( # Renamed for clarity, as it's a description not the patch itself
            "Add support for specifying servers in the generated OpenAPI schema by accepting a 'servers' parameter in the application initialization."
        ),
         # It's helpful to provide the correct patch if known, for better review context
        "correct_patch_example": (
        "diff --git a/fastapi/applications.py b/fastapi/applications.py\nindex 3306aab3d95eb..c21087911ebf4 100644\n--- a/fastapi/applications.py\n+++ b/fastapi/applications.py\n@@ -38,6 +38,7 @@ def __init__(\n         version: str = \"0.1.0\",\n         openapi_url: Optional[str] = \"/openapi.json\",\n         openapi_tags: Optional[List[Dict[str, Any]]] = None,\n+        servers: Optional[List[Dict[str, Union[str, Any]]]] = None,\n         default_response_class: Type[Response] = JSONResponse,\n         docs_url: Optional[str] = \"/docs\",\n         redoc_url: Optional[str] = \"/redoc\",\n@@ -70,6 +71,7 @@ def __init__(\n         self.title = title\n         self.description = description\n         self.version = version\n+        self.servers = servers\n         self.openapi_url = openapi_url\n         self.openapi_tags = openapi_tags\n         # TODO: remove when discarding the openapi_prefix parameter\n@@ -106,6 +108,7 @@ def openapi(self, openapi_prefix: str = \"\") -> Dict:\n                 routes=self.routes,\n                 openapi_prefix=openapi_prefix,\n                 tags=self.openapi_tags,\n+                servers=self.servers,\n             )\n         return self.openapi_schema\n \ndiff --git a/fastapi/openapi/models.py b/fastapi/openapi/models.py\nindex a7c4460fab43a..13dc59f189527 100644\n--- a/fastapi/openapi/models.py\n+++ b/fastapi/openapi/models.py\n@@ -63,7 +63,7 @@ class ServerVariable(BaseModel):\n \n \n class Server(BaseModel):\n-    url: AnyUrl\n+    url: Union[AnyUrl, str]\n     description: Optional[str] = None\n     variables: Optional[Dict[str, ServerVariable]] = None\n \ndiff --git a/fastapi/openapi/utils.py b/fastapi/openapi/utils.py\nindex b6221ca202826..5a0c89a894cb3 100644\n--- a/fastapi/openapi/utils.py\n+++ b/fastapi/openapi/utils.py\n@@ -86,7 +86,7 @@ def get_openapi_security_definitions(flat_dependant: Dependant) -> Tuple[Dict, L\n def get_openapi_operation_parameters(\n     *,\n     all_route_params: Sequence[ModelField],\n-    model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str]\n+    model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str],\n ) -> List[Dict[str, Any]]:\n     parameters = []\n     for param in all_route_params:\n@@ -112,7 +112,7 @@ def get_openapi_operation_parameters(\n def get_openapi_operation_request_body(\n     *,\n     body_field: Optional[ModelField],\n-    model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str]\n+    model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str],\n ) -> Optional[Dict]:\n     if not body_field:\n         return None\n@@ -318,12 +318,15 @@ def get_openapi(\n     description: str = None,\n     routes: Sequence[BaseRoute],\n     openapi_prefix: str = \"\",\n-    tags: Optional[List[Dict[str, Any]]] = None\n+    tags: Optional[List[Dict[str, Any]]] = None,\n+    servers: Optional[List[Dict[str, Union[str, Any]]]] = None,\n ) -> Dict:\n     info = {\"title\": title, \"version\": version}\n     if description:\n         info[\"description\"] = description\n     output: Dict[str, Any] = {\"openapi\": openapi_version, \"info\": info}\n+    if servers:\n+        output[\"servers\"] = servers\n     components: Dict[str, Dict] = {}\n     paths: Dict[str, Dict] = {}\n     flat_models = get_flat_models_from_routes(routes)\n",

        )
    }
    return input_data

def query_llm_for_review(patch: str, input_data: dict) -> str:
    """
    Queries the LLM to get a review of the provided patch using Google's Generative AI (Gemini).
    Provides context about the original problem.
    """
    prompt = f"""You are an expert code reviewer.
    Analyze the provided "Bad Patch" which was intended to solve the "Problem Statement".
    The goal was to: {input_data.get('desired_patch_description', 'No description provided.')}

    Explain specifically why the "Bad Patch" is incorrect, incomplete, or fails to address the "Problem Statement" adequately. Focus on the functional correctness and relevance to the problem.

    Problem Statement:
    {input_data.get('problem_statement', 'No problem statement provided.')}

    (Optional Reference - Correct Patch Snippet):
    ```diff
    {input_data.get('correct_patch_example', 'N/A')}"""
    
    # --- Use the current API ---
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create GenerationConfig object for parameters
        config = GenerationConfig(
            temperature=0.3, # Slightly higher than 0.2 for a bit more variance if needed
            max_output_tokens=500
        )

        # Generate content using the model instance
        response = model.generate_content(prompt, generation_config=config)

        # Extract the text, handling potential errors or blocks
        review = response.text.strip()

    except ValueError as e:
        # Handle potential errors like blocked content more gracefully
        review = f"Review generation failed or content blocked. Details: {e}\nResponse feedback: {getattr(response, 'prompt_feedback', 'N/A')}"
    except Exception as e:
        # Catch other potential exceptions during API call or processing
        review = f"An unexpected error occurred during review generation: {e}"

    if not review: # Handle empty reviews
        review = "No review generated or review was empty."

    return review
    # # Use the text generation function from the Google Generative AI library.
    # # Here we use 'text-bison-001' as the model. You may adjust the model name and parameters as needed.
    # response = genai.generate_text(
    #     prompt=prompt,
    #     model="text-bison-001",
    #     temperature=0.2,        # Lower temperature for more deterministic feedback
    #     max_output_tokens=500
    # )
    
    # # Extract the output from the response. The response contains a list of candidates.
    # if response and response.candidates:
    #     review = response.candidates[0].output.strip()
    # else:
    #     review = "No review generated."
    # return review
def main():
    # Ensure your API key is set in the environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: Please set your GOOGLE_API_KEY environment variable.")
        print("You can get a key from Google AI Studio: https://aistudio.google.com/app/apikey")
        return # Exit gracefully if key is missing

    # Configure the google.generativeai library with your API key.
    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        print(f"Error configuring Google Generative AI: {e}")
        return

    # Process input data (simulate gathering data from a pull request or issue)
    input_data = process_input_data()

    # Generate a patch that is likely to be bad (i.e., doesn't actually fix the problem)
    bad_patch = (
        "diff --git a/fastapi/dependencies/utils.py b/fastapi/dependencies/utils.py\n"
        "index 1a660f5d355fa..15e7790cea9b5 100644\n"
        "--- a/fastapi/dependencies/utils.py\n"
        "+++ b/fastapi/dependencies/utils.py\n"
        "@@ -478,6 +478,7 @@ async def solve_dependencies(\n"
        "                 name=sub_dependant.name,\n" # Corrected indentation based on likely context
        "                 security_scopes=sub_dependant.security_scopes,\n" # Corrected indentation
        "             )\n" # Corrected indentation
        "+            use_sub_dependant.name = sub_dependant.name\n" # Added '+' assuming this was the intended change line
        " \n"
        "         solved_result = await solve_dependencies(\n"
        "             request=request,"
    )
    print("--- Bad Patch ---")
    print(bad_patch)
    print("-" * 27)

    # Query the LLM for a review of the generated patch, providing context
    print("\nQuerying LLM for review...")
    review = query_llm_for_review(bad_patch, input_data)
    print("\n--- LLM Review ---")
    print(review)
    print("-" * 18)

if __name__ == "__main__":
    main()
