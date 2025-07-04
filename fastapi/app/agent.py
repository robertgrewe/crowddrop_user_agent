from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv
import yaml
import json
import os
import requests
import urllib.parse
import subprocess
import asyncio



from services import CrowdDropServices  # Import the class

load_dotenv(find_dotenv())

# Load authentication details from environment variables

# GITHUB API
API_GITHUB_KEY = os.getenv("API_GITHUB_KEY")
API_GITHUB_BASE_URL = os.getenv("API_GITHUB_BASE_URL")

# CrowdDrop API
API_BASE_URL = os.getenv("API_BASE_URL")
API_AUTH_URL = os.getenv("API_AUTH_URL")
API_OPENAPI_SPEC_URL = os.getenv("API_OPENAPI_SPEC_URL")
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME")
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD")
CROWDDROP_MCP_SERVER = os.getenv("CrowdDrop_MCP_SERVER")

if not CROWDDROP_USERNAME or not CROWDDROP_PASSWORD or not API_GITHUB_KEY or not API_GITHUB_BASE_URL:
    raise ValueError("Missing API credentials or endpoint in environment variables.")

def authenticate_and_get_token(api_base_url, username, password):
    """
    ***CRITICAL:*** Replace this with your API's *actual* authentication method.
    This is a placeholder for demonstration purposes only.
    """

    # URL-encode the username and password
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

    # local model running in docker container
    # llm = OllamaLLM(
    #     model="llama3", 
    #     temperature=0.0, 
    #     max_tokens=None, 
    #     timeout=None, 
    #     max_retries=2, 
    #     base_url="http://localhost:11434",
    #     verbose=True)

    # llm = LlamaCpp(
    #     model_path=LLAMA_MODEL_PATH,
    #     temperature=0.0,
    #     n_ctx=2048,  # Adjust context window as needed
    #     n_gpu_layers=1,  # Use GPU if available
    # )

    openapi_spec_url = API_OPENAPI_SPEC_URL
    try:
        response = requests.get(openapi_spec_url)
        response.raise_for_status()
        raw_openai_api_spec = response.json()

        # --- IMPORTANT MODIFICATION HERE ---
        # Ensure 'servers' key exists and is a list
        if "servers" not in raw_openai_api_spec or not isinstance(raw_openai_api_spec["servers"], list):
            raw_openai_api_spec["servers"] = []

        # Check if the desired base URL is already present to avoid duplicates
        # We need to ensure the full base URL is there.
        # Since your current spec has '/app', let's replace it or add the full one.
        
        # Option A: Replace if '/app' exists, otherwise add the full URL
        found_full_url = False
        for i, server in enumerate(raw_openai_api_spec["servers"]):
            if server.get("url") == "/app":
                raw_openai_api_spec["servers"][i]["url"] = API_BASE_URL + "/app"
                found_full_url = True
                break
        if not found_full_url:
            raw_openai_api_spec["servers"].insert(0, {"url": API_BASE_URL + "/app"}) # Add it at the beginning

        # You might also want to ensure the root API_BASE_URL is present if some endpoints don't use /app
        # if not any(s.get("url") == API_BASE_URL for s in raw_openai_api_spec["servers"]):
        #     raw_openai_api_spec["servers"].insert(0, {"url": API_BASE_URL})

        print(f"Modified OpenAPI Spec Servers: {raw_openai_api_spec.get('servers')}")

        openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
        print("Successfully loaded and modified OpenAPI spec from URL.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OpenAPI spec from URL: {e}")
        exit()

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

    # --------

    # Launch docker container with MCP server

    #url = "http://host.docker.internal:8000/mcp"
    #url = "http://127.0.0.1:8000/mcp"
    url = "http://localhost:8000/mcp"
    #url = "localhost:8000/mcp"


    mcp_client = MultiServerMCPClient({
        "local": {
            "transport": "sse",
            "url": url,  # exact SSE endpoint URL here
            #"timeout": 20.0  # increase if supported
        },
        # "crowddrop": {
        #     "transport": "sse",
        #     "url": CROWDDROP_MCP_SERVER,  # URL for the Crowddrop MCP server
        # },
    })

    async def load_tools():
        tools = await mcp_client.get_tools()
        return tools

    tools = asyncio.run(load_tools())

    async def main():
        agent = initialize_agent(
            tools,  # tools loaded from MultiServerMCPClient
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

        user_query = "What tools are available?"
        response = await agent.ainvoke({"input": user_query})
        print(f"response: {response}")

        user_query = "Add 10 to 5 and return the result."
        response = await agent.ainvoke({"input": user_query})
        print(f"response: {response}")

        user_query = "List all CrowdDrop operations."
        response = await agent.ainvoke({"input": user_query})
        print(f"response: {response}")

    # Directly invoke the async function
    #asyncio.run(main())

    # --------

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
    #user_query = f"Get task with id=6867d69a820a34545cb58224."
    user_query = f"Work on task with the title Take image of this tree."
    #user_query = f"Use the work_on endpoint to start workong on task id 6867d69a820a34545cb58224."

    response = openapi_agent.invoke(user_query)
    print(f"response: {response}")
