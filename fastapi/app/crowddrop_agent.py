# crowddrop_agent.py
"""
This file contains functions and logic specifically related to the CrowdDrop API integration.
It encapsulates:
1.  Authentication with the CrowdDrop API to obtain an access token.
2.  Initialization of a LangChain OpenAPI sub-agent tailored to interact with the CrowdDrop API's OpenAPI specification.
3.  A utility function to create the LangChain Tool object for the CrowdDrop API, which can then be used by the main agent.

By centralizing CrowdDrop-specific logic here, the main agent.py file remains cleaner and focuses on the overall hierarchical agent orchestration.
"""

import os
import json
import requests
import urllib.parse
import subprocess
import logging
import asyncio
from typing import Optional, Any, Tuple
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from dotenv import load_dotenv, find_dotenv

# Load environment variables at the very beginning of the module
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# Load environment variables specific to CrowdDrop API
# IMPORTANT: Provide default values for local testing if .env isn't loaded
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8003")
# Derived default for API_AUTH_URL, ensuring it uses the (potentially defaulted) API_BASE_URL
API_AUTH_URL = os.getenv("API_AUTH_URL", f"{API_BASE_URL.rstrip('/')}/auth/login")
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME", "testuser") # Added default for testing
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD", "testpassword") # Added default for testing
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini") # Added default for testing

# Aggressively correct API_BASE_URL immediately after loading it
if API_BASE_URL:
    API_BASE_URL = API_BASE_URL.replace("dev.crowddot.aidobotics.ai", "dev.crowddrop.aidobotics.ai")
    logger.info(f"Corrected API_BASE_URL to: {API_BASE_URL}")
else:
    logger.error("API_BASE_URL is not set. Please ensure it's configured in your .env file or environment.")

# Ensure API_OPENAPI_SPEC_URL is derived from the corrected API_BASE_URL
API_OPENAPI_SPEC_URL = API_BASE_URL.rstrip('/') + "/openapi.json"


# DEBUGGING: Print the API_BASE_URL and API_OPENAPI_SPEC_URL immediately after they are loaded
logger.debug(f"DEBUG: API_BASE_URL loaded in crowddrop_agent.py: {API_BASE_URL}")
logger.debug(f"DEBUG: API_OPENAPI_SPEC_URL loaded in crowddrop_agent.py: {API_OPENAPI_SPEC_URL}")


def authenticate_and_get_token(username: str, password: str) -> Optional[str]:
    """
    Authenticates with the API and retrieves an ID token.
    ***CRITICAL:*** Replace this with your API's *actual* authentication method.
    This is a placeholder for demonstration purposes only.
    """
    encoded_username = urllib.parse.quote(username)
    encoded_password = urllib.parse.quote(password)
    full_url = f"{API_AUTH_URL}?username={encoded_username}&password={encoded_password}"

    curl_command = [
        "curl",
        "-X", "POST",
        full_url,
        "-H", "accept: application/json",
        "-d", "",
    ]

    try:
        process = subprocess.Popen(curl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Authentication request failed (curl): {stderr.decode()}")
            return None

        response_data = json.loads(stdout.decode())
        logger.debug(f"Authentication response data: {response_data}")
        return response_data.get("id_token")

    except Exception as e:
        logger.exception(f"Authentication request failed (exception): {e}")
        return None

async def initialize_crowddrop_sub_agent() -> Tuple[Any, Optional[str], Any]:
    """
    Initializes the CrowdDrop OpenAPI sub-agent and performs initial authentication.
    Returns the sub-agent executor, the access token, and the LLM instance used.
    """
    logger.info("Initializing CrowdDrop OpenAPI sub-agent...")

    # Authenticate and get token
    access_token = authenticate_and_get_token(CROWDDROP_USERNAME, CROWDDROP_PASSWORD)
    if access_token is None:
        logger.error("Authentication failed. Cannot initialize CrowdDrop sub-agent without token.")
        return None, None, None

    # Load and modify OpenAPI spec
    openapi_spec_url = API_OPENAPI_SPEC_URL
    if not openapi_spec_url:
        logger.error("API_OPENAPI_SPEC_URL is not set. Cannot fetch OpenAPI spec.")
        return None, access_token, None

    try:
        response = requests.get(openapi_spec_url)
        response.raise_for_status()
        raw_openai_api_spec = response.json()

        # AGGRESSIVE CORRECTION: Convert to string, replace typo, convert back to JSON
        spec_string = json.dumps(raw_openai_api_spec)
        corrected_spec_string = spec_string.replace("dev.crowddot.aidobotics.ai", "dev.crowddrop.aidobotics.ai")
        raw_openai_api_spec = json.loads(corrected_spec_string)
        logger.info("Performed aggressive typo correction on OpenAPI spec string.")

        # Ensure the 'servers' list is explicitly set to the correct API_BASE_URL
        raw_openai_api_spec["servers"] = [{"url": API_BASE_URL.rstrip('/') + "/app"}]

        # NEW DEBUG PRINT: Print the raw_openai_api_spec after all modifications
        logger.debug(f"DEBUG: Raw OpenAPI Spec AFTER all modifications (first 500 chars): {str(raw_openai_api_spec)[:500]}...")
        logger.debug(f"DEBUG: Raw OpenAPI Spec Servers AFTER all modifications: {raw_openai_api_spec.get('servers')}")


        logger.info(f"Modified OpenAPI Spec Servers: {raw_openai_api_spec.get('servers')}")
        openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
        logger.info("Successfully loaded and modified OpenAPI spec from URL.")

        # NEW DEBUG PRINT: Check the reduced spec
        logger.debug(f"DEBUG: Reduced OpenAPI Spec (first 500 chars): {str(openai_api_spec)[:500]}...")
        # If you want to see the full reduced spec, uncomment the following lines:
        # with open("debug_reduced_openapi_spec.json", "w") as f:
        #     # You might need to convert openai_api_spec to a dict if it's an object
        #     # For OpenAPISpec object, you might need openai_api_spec.dict() or similar
        #     json.dump(openai_api_spec.dict() if hasattr(openai_api_spec, 'dict') else str(openai_api_spec), f, indent=2)


    except requests.exceptions.RequestException as e:
        logger.exception(f"Error fetching OpenAPI spec from URL: {e}")
        return None, access_token, None
    except json.JSONDecodeError as e:
        logger.exception(f"Error decoding/encoding OpenAPI spec JSON during correction: {e}")
        return None, access_token, None

    # Initialize the LLM based on the selected model provider
    logger.info(f"Initializing LLM for model provider: {MODEL_PROVIDER}")

    llm = None

    if MODEL_PROVIDER == "github":
        logger.info("Initializing GitHub OpenAI...")
        API_GITHUB_KEY = os.getenv("API_GITHUB_KEY")
        API_GITHUB_BASE_URL = os.getenv("API_GITHUB_BASE_URL")
        API_GITHUB_MODEL = os.getenv("API_GITHUB_MODEL", "gpt-4o-mini")

        if not API_GITHUB_KEY or not API_GITHUB_MODEL:
            logger.error("GitHub OpenAI API key or model name not set in .env for OpenAI provider.")
            raise ValueError("GitHub OpenAI API key or model name not set in .env for OpenAI provider.")

        llm = ChatOpenAI(
            model_name=API_GITHUB_MODEL,
            temperature=0.0,
            api_key=API_GITHUB_KEY,
            base_url=API_GITHUB_BASE_URL,
        )
        logger.info(f"GitHub OpenAI initialized with model: {API_GITHUB_MODEL}")

    elif MODEL_PROVIDER == "gemini":
        logger.info("Initializing ChatGoogleGenerativeAI...")
        API_GEMINI_KEY = os.getenv("API_GEMINI_KEY")
        API_GEMINI_MODEL = os.getenv("API_GEMINI_MODEL", "gemini-1.5-flash")

        if not API_GEMINI_KEY or not API_GEMINI_MODEL:
            logger.error("Gemini API key or model name not set in .env for Gemini provider.")
            raise ValueError("Gemini API key or model name not set in .env for Gemini provider.")

        llm = ChatGoogleGenerativeAI(
            model=API_GEMINI_MODEL,
            temperature=0.0,
            google_api_key=API_GEMINI_KEY,
        )
        logger.info(f"ChatGoogleGenerativeAI initialized with model: {API_GEMINI_MODEL}")

    else:
        logger.critical(f"Invalid MODEL_PROVIDER specified in .env: {MODEL_PROVIDER}. Must be 'github' or 'gemini'.")
        raise ValueError(f"Invalid MODEL_PROVIDER specified in .env: {MODEL_PROVIDER}. Must be 'github' or 'gemini'.")
    logger.info("LLM initialized successfully.")

    # Configure RequestsWrapper with the Authorization header
    openapi_requests_wrapper = RequestsWrapper(
        headers={"Authorization": f"Bearer {access_token}", "accept": "application/json"}
    )

    # Create the OpenAPI agent executor
    crowddrop_openapi_agent_executor = planner.create_openapi_agent(
        api_spec=openai_api_spec,
        llm=llm,
        requests_wrapper=openapi_requests_wrapper,
        tools=[],
        verbose=True,
        allow_dangerous_requests=True,
    )
    logger.info("CrowdDrop OpenAPI sub-agent created successfully.")
    return crowddrop_openapi_agent_executor, access_token, llm

def get_crowddrop_api_tool(crowddrop_openapi_agent_executor_instance: Any) -> Tool:
    """
    Returns the Tool for interacting with the CrowdDrop API.
    """
    return Tool(
        name="crowddrop_api_interface",
        func=crowddrop_openapi_agent_executor_instance.invoke,
        description=(
            "A powerful tool for interacting with the CrowdDrop API. "
            "Use this tool for all tasks related to CrowdDrop, such as listing tasks, "
            "working on tasks, completing tasks, querying task details (including coordinates), "
            "or **updating the humanoid robot's own location**. "
            "Pass your query directly to this tool, for example: "
            "'crowddrop_api_interface(\"list all tasks near me\")', "
            "'crowddrop_api_interface(\"query details for task 123\")', or "
            "'crowddrop_api_interface(\"update my location to latitude 52.123 and longitude 13.456\")'."
            "This tool understands natural language queries related to CrowdDrop API operations."
        ),
    )

if __name__ == "__main__":
    # Configure basic logging for local testing if not already configured by import
    if not logging.root.handlers:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def test_crowddrop_agent_initialization():
        logger.info("\n--- Testing crowddrop_agent.py initialization ---")

        # Test authentication function
        logger.info("Attempting to authenticate and get token...")
        token = authenticate_and_get_token(CROWDDROP_USERNAME, CROWDDROP_PASSWORD)
        if token:
            logger.info(f"Authentication successful. Token (first 10 chars): {token[:10]}...")
        else:
            logger.error("Authentication failed during local test.")

        # Test sub-agent initialization
        logger.info("Attempting to initialize CrowdDrop sub-agent...")
        sub_agent_executor, sub_access_token, sub_llm = await initialize_crowddrop_sub_agent()

        if sub_agent_executor:
            logger.info("CrowdDrop sub-agent initialized successfully.")
            logger.info(f"Sub-agent LLM type: {type(sub_llm).__name__}")
            logger.info(f"Sub-agent access token (first 10 chars): {sub_access_token[:10]}...")

            # Test the crowddrop_api_tool by invoking the executor directly
            logger.info("\nAttempting to invoke crowddrop_api_interface via sub-agent executor (mocked if API is not live)...")
            try:
                # This will actually try to hit the API_BASE_URL/openapi.json and then the API endpoints
                # If your API is not running, this will likely fail with a connection error.
                # For a truly isolated test, you'd mock the requests.get and the API endpoints.
                # But for testing the LangChain setup, trying to connect is valid.
                # RE-ADDED: output_instructions to help agent parse the JSON
                dummy_query_result = await sub_agent_executor.ainvoke({"input": "list all tasks", "output_instructions": "Extract all tasks from the response."})
                logger.info(f"CrowdDrop API tool invocation result (partial): {str(dummy_query_result)[:200]}...")
            except Exception as e:
                logger.error(f"Error during crowddrop_api_interface invocation: {e}")
                logger.warning("If the CrowdDrop API is not running, this error is expected.")
        else:
            logger.error("Failed to initialize CrowdDrop sub-agent.")

        logger.info("\n--- End of crowddrop_agent.py initialization test ---")

    asyncio.run(test_crowddrop_agent_initialization())
