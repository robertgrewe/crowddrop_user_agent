from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain.tools import Tool
from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv
import yaml
import json
import os
import requests
import urllib.parse
import subprocess

from fastapi.app.services import CrowdDropServices  # Import the class

load_dotenv(find_dotenv())

# Load authentication details from environment variables

# GITHUB API
API_GITHUB_KEY = os.getenv("API_GITHUB_KEY")
API_GITHUB_BASE_URL = os.getenv("API_GITHUB_BASE_URL")

# CrowdDrop API
API_BASE_URL = os.getenv("API_BASE_URL")
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME")
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD")

if not CROWDDROP_USERNAME or not CROWDDROP_PASSWORD or not API_GITHUB_KEY or not API_GITHUB_BASE_URL:
    raise ValueError("Missing API credentials or endpoint in environment variables.")

def authenticate_and_get_token(api_base_url, username, password):
    """
    ***CRITICAL:*** Replace this with your API's *actual* authentication method.
    This is a placeholder for demonstration purposes only.
    """
    auth_url = f"{api_base_url}/app/auth/login"

    # URL-encode the username and password
    encoded_username = urllib.parse.quote(username)
    encoded_password = urllib.parse.quote(password)

    full_url = f"{auth_url}?username={encoded_username}&password={encoded_password}"

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
            print(f"Authentication request failed (curl): {stderr.decode()}")
            return None

        response_data = json.loads(stdout.decode())
        print(f"response_data: {response_data}")
        return response_data.get("id_token")

    except Exception as e:
        print(f"Authentication request failed (exception): {e}")
        return None

def make_api_request(api_base_url, endpoint, method, headers=None, params=None, data=None, token=None):
    """
    Generic function to make API requests with authentication.
    """
    url = f"{api_base_url}{endpoint}"
    try:
        headers = headers or {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        headers['accept'] = 'application/json'
        print(f"Request Headers: {headers}") #Added line
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, data=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=headers, data=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers, data=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

@tool
def crowddrop_get_tasks(crowddrop_service: CrowdDropServices, page: int = 1, size: int = 10) -> str:
    """
    Retrieves tasks from the Crowddrop API with pagination.

    Args:
        crowddrop_service: An instance of the CrowdDropServices class.
        page: The page number to retrieve.
        size: The number of tasks per page.
    """
    if crowddrop_service.token is None:
        if not crowddrop_service.authenticate():
            return "Authentication failed."
    tasks = crowddrop_service.get_tasks(page, size)
    if tasks:
        return json.dumps(tasks, indent=2)
    else:
        return "Failed to retrieve tasks."

if __name__ == "__main__":
    """
    This script will ask a question to the OpenAPI agent for Crowddrop API.
    """

# for models on GitHub: https://github.com/marketplace/models/azureml-meta/Llama-3-3-70B-Instruct/playground

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        #model_name="gpt-4o",
        temperature=0.0,
        api_key=API_GITHUB_KEY,
        base_url=API_GITHUB_BASE_URL,
    )

    # llm = LlamaCpp(
    #     model_path=LLAMA_MODEL_PATH,
    #     temperature=0.0,
    #     n_ctx=2048,  # Adjust context window as needed
    #     n_gpu_layers=1,  # Use GPU if available
    # )

    with open("./crowddrop_openapi.yaml") as f:
        raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)
        openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)

    access_token = authenticate_and_get_token(API_BASE_URL, CROWDDROP_USERNAME, CROWDDROP_PASSWORD)
    if access_token is None:
        print("Authentication failed. Exiting.")
        exit()

    tools = [
#         Tool(
#             name="api_request",
#             func=lambda input_dict: make_api_request(
#                 API_BASE_URL,
#                 endpoint=input_dict.get("endpoint"),
#                 method=input_dict.get("method"),
#                 headers={"Authorization": f"Bearer {access_token}"} if access_token else None,
#                 params=input_dict.get("params"),
#                 data=input_dict.get("data"),
#                 token=access_token,
#             ),
#             description="Use this tool to make authorized API requests. Include the 'Authorization: Bearer <token>' header using the provided token for endpoints that require authentication. The input must be a JSON object containing the keys 'endpoint', 'method', 'params' (optional), and 'data' (optional)."
#             "An equivalent curl call would look like this: curl -X 'GET' \
#   'https://dev.crowddrop.aidobotics.ai/app/tasks/?page=1&size=10' \
#   -H 'accept: application/json' \
#   -H 'Authorization: Bearer <token>'",
#         )
    ]

    # Configure RequestsWrapper with the Authorization header
    headers = {"Authorization": f"Bearer {access_token}", "accept": "application/json"}
    requests_wrapper = RequestsWrapper(headers=headers)

    openapi_agent = planner.create_openapi_agent(
        api_spec=openai_api_spec,
        llm=llm,
        requests_wrapper=requests_wrapper,
        tools=tools,
        verbose=True,
        allow_dangerous_requests=True,
    )


    #user_query = f"Get task with id=67b8760e920af4b7a5ba837f. Always include the Authorization header with Bearer {access_token} and accept: application/json when making API calls. Before making an API call always print out the equivalent curl command."
    #user_query = f"List all tasks. Always include the Authorization header with Bearer {access_token} when making API calls."
    user_query = f"Get task with id=67b8760e920af4b7a5ba837f."
    response = openapi_agent.invoke(user_query)
    print(f"response: {response}")