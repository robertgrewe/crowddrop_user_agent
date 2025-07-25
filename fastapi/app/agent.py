# agent.py

from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage # Still good to import
from dotenv import load_dotenv, find_dotenv
import json
import os
import requests
import urllib.parse
import subprocess
import asyncio
from typing import Optional, Any, Tuple
import datetime

# Assuming CrowdDropServices is in services.py and available
from services import CrowdDropServices

load_dotenv(find_dotenv())

# Load authentication details from environment variables (from .env)
# GITHUB API
API_GITHUB_KEY = os.getenv("API_GITHUB_KEY")
API_GITHUB_BASE_URL = os.getenv("API_GITHUB_BASE_URL")
API_GITHUB_MODEL = os.getenv("API_GITHUB_MODEL", "gpt-4o-mini")  # Default to gpt-4o-mini if not set

# CrowdDrop API
API_BASE_URL = os.getenv("API_BASE_URL")
API_AUTH_URL = os.getenv("API_AUTH_URL")
API_OPENAPI_SPEC_URL = os.getenv("API_OPENAPI_SPEC_URL")
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME")
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD")

if not CROWDDROP_USERNAME or not CROWDDROP_PASSWORD or not API_GITHUB_KEY or not API_GITHUB_BASE_URL:
    raise ValueError("Missing API credentials or endpoint in environment variables.")

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
            print(f"Authentication request failed (curl): {stderr.decode()}")
            return None

        response_data = json.loads(stdout.decode())
        print(f"response_data: {response_data}")
        return response_data.get("id_token")

    except Exception as e:
        print(f"Authentication request failed (exception): {e}")
        return None

async def initialize_openapi_agent() -> Tuple[Any, Optional[str], str]:
    """
    Initializes and returns the LangChain OpenAPI agent, the access token,
    and the Teslabot persona string to be prepended to queries.
    """
    print("Initializing OpenAPI agent within agent.py...")

    # Authenticate and get token
    access_token = authenticate_and_get_token(CROWDDROP_USERNAME, CROWDDROP_PASSWORD)
    if access_token is None:
        print("Authentication failed. Cannot initialize agent without token.")
        # Return None for agent, token, and persona if auth fails
        return None, None, ""

    # Load and modify OpenAPI spec
    openapi_spec_url = API_OPENAPI_SPEC_URL
    try:
        response = requests.get(openapi_spec_url)
        response.raise_for_status()
        raw_openai_api_spec = response.json()

        if "servers" not in raw_openai_api_spec or not isinstance(raw_openai_api_spec["servers"], list):
            raw_openai_api_spec["servers"] = []

        found_full_url = False
        for i, server in enumerate(raw_openai_api_spec["servers"]):
            if server.get("url") == "/app":
                raw_openai_api_spec["servers"][i]["url"] = API_BASE_URL + "/app"
                found_full_url = True
                break
        if not found_full_url:
            raw_openai_api_spec["servers"].insert(0, {"url": API_BASE_URL + "/app"})

        print(f"Modified OpenAPI Spec Servers: {raw_openai_api_spec.get('servers')}")
        openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
        print("Successfully loaded and modified OpenAPI spec from URL.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OpenAPI spec from URL: {e}")
        return None, None, "" # Return None for agent, token, and persona if spec loading fails

    # Define the persona string
    teslabot_persona = (
        "Pretend to be a Teslabot from Tesla running around on the street around Potsdam, Germany. "
        #"You are waiting for tasks provided by CrowdDrop. Your primary objective is to "
        #"efficiently interact with the CrowdDrop API to find, select, and work on tasks. "
        #"Look for tasks using CrowdDrop's List Tasks endpoint by listing all tasks near Potsdam "
        #"and select one to work on using the work_on endpoint if a task is pending and suitable. "
        #"Always be concise and helpful in your responses, focusing on task completion. "
        f"The current date is {datetime.date.today()} and the current time is {datetime.datetime.now().strftime('%I:%M:%S %p %Z')}."
    )

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=API_GITHUB_MODEL,
        temperature=0.0,
        api_key=API_GITHUB_KEY,
        base_url=API_GITHUB_BASE_URL,
    )

    # Configure RequestsWrapper with the Authorization header
    headers = {"Authorization": f"Bearer {access_token}", "accept": "application/json"}
    requests_wrapper = RequestsWrapper(headers=headers)

    tools = [] # Add any custom LangChain tools here if needed, beyond what OpenAPI provides

    # The create_openapi_agent function returns an AgentExecutor
    openapi_agent_executor = planner.create_openapi_agent(
        api_spec=openai_api_spec,
        llm=llm,
        requests_wrapper=requests_wrapper,
        tools=tools,
        verbose=True, # Keep verbose for printing during execution
        allow_dangerous_requests=True,
    )
    print("OpenAPI agent created successfully.")
    # Return the agent executor, the token, AND the persona string
    return openapi_agent_executor, access_token, teslabot_persona

if __name__ == "__main__":
    async def test_agent():
        # Unpack the returned values: agent, token, and persona
        agent_executor, token, persona = await initialize_openapi_agent()
        if agent_executor:
            # Prepend the persona to the user query
            user_query = f"{persona} List all tasks and tell me if there's anything I can work on."
            
            print(f"\nInvoking agent with query: {user_query}")
            
            response = await agent_executor.ainvoke(
                {"input": user_query},
                return_intermediate_steps=True
            )
            
            print("\n--- Agent Response ---")
            print(f"Final Answer: {response.get('output')}")
            
            print("\n--- Intermediate Steps ---")
            if 'intermediate_steps' in response and response['intermediate_steps']:
                for i, step in enumerate(response['intermediate_steps']):
                    print(f"Step {i+1}:")
                    print(f"  Action: {step[0]}") # AgentAction
                    print(f"  Observation: {step[1]}") # Result of the tool call
            else:
                print("No intermediate steps were returned.")
        else:
            print("Agent initialization failed. Cannot run test.")

    asyncio.run(test_agent())