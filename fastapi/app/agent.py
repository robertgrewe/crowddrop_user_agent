# agent.py

from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, StructuredTool # Keep these for other potential generic tools
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv, find_dotenv
import json
import os
import requests
import urllib.parse
import subprocess
import asyncio
from typing import Optional, Any, Tuple, List, Dict
import datetime

# We don't necessarily need CrowdDropServices from services.py for this approach
# as the OpenAPI agent will directly handle API interactions.
# from services import CrowdDropServices # You can remove this import if not used elsewhere

load_dotenv(find_dotenv())

# Load authentication details from environment variables (from .env)

# Get the model provider from the environment variable
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

# CrowdDrop API
API_BASE_URL = os.getenv("API_BASE_URL")
API_AUTH_URL = os.getenv("API_AUTH_URL")
API_OPENAPI_SPEC_URL = os.getenv("API_OPENAPI_SPEC_URL")
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME")
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD")

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

async def initialize_hierarchical_agent() -> Tuple[Any, Optional[str], str]:
    """
    Initializes and returns the LangChain hierarchical agent, the access token,
    and the Teslabot persona string.
    """
    print("Initializing hierarchical agent within agent.py...")

    # Authenticate and get token
    access_token = authenticate_and_get_token(CROWDDROP_USERNAME, CROWDDROP_PASSWORD)
    if access_token is None:
        print("Authentication failed. Cannot initialize agent without token.")
        return None, None, ""

    # --- PART 1: Initialize the specialized CrowdDrop OpenAPI agent (the sub-agent) ---
    print("Initializing CrowdDrop OpenAPI sub-agent...")
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
        return None, None, "" # Return None if spec loading fails

    # Initialize the LLM based on the selected model provider
    print(f"Initializing LLM for model provider: {MODEL_PROVIDER}")

    llm = None # Initialize llm to None

    if MODEL_PROVIDER == "github":

        # Initialize GitHub OpenAI LLM
        print("Initializing GitHub OpenAI...")
        # Retrieve GitHub OpenAI specific variables
        API_GITHUB_KEY = os.getenv("API_GITHUB_KEY")
        API_GITHUB_BASE_URL = os.getenv("API_GITHUB_BASE_URL")
        API_GITHUB_MODEL = os.getenv("API_GITHUB_MODEL", "gpt-4o-mini")

        if not API_GITHUB_KEY or not API_GITHUB_MODEL:
            raise ValueError("GitHub OpenAI API key or model name not set in .env for OpenAI provider.")

        llm = ChatOpenAI(
            model_name=API_GITHUB_MODEL,
            temperature=0.0,
            api_key=API_GITHUB_KEY,
            base_url=API_GITHUB_BASE_URL,
        )
        print(f"GitHub OpenAI initialized with model: {API_GITHUB_MODEL}")

    elif MODEL_PROVIDER == "gemini":

        # Initialize Gemini LLM
        print("Initializing ChatGoogleGenerativeAI...")
        # Retrieve Gemini specific variables
        API_GEMINI_KEY = os.getenv("API_GEMINI_KEY")
        API_GEMINI_MODEL = os.getenv("API_GEMINI_MODEL", "gemini-2.5-flash")

        if not API_GEMINI_KEY or not API_GEMINI_MODEL:
            raise ValueError("Gemini API key or model name not set in .env for Gemini provider.")

        llm = ChatGoogleGenerativeAI(
            model=API_GEMINI_MODEL,
            temperature=0.0,
            google_api_key=API_GEMINI_KEY,
        )
        print(f"ChatGoogleGenerativeAI initialized with model: {API_GEMINI_MODEL}")

    else:
        raise ValueError(f"Invalid MODEL_PROVIDER specified in .env: {MODEL_PROVIDER}. Must be 'github' or 'gemini'.")
    print("LLM initialized successfully.")

    # Configure RequestsWrapper with the Authorization header for the OpenAPI agent
    openapi_requests_wrapper = RequestsWrapper(headers={"Authorization": f"Bearer {access_token}", "accept": "application/json"})

    # Create the OpenAPI agent executor
    # This agent will know how to interact with the CrowdDrop API via its OpenAPI spec
    crowddrop_openapi_agent_executor = planner.create_openapi_agent(
        api_spec=openai_api_spec,
        llm=llm,
        requests_wrapper=openapi_requests_wrapper,
        tools=[], # No additional tools for this sub-agent, it uses the OpenAPI spec itself
        verbose=True, # Keep verbose for seeing sub-agent's thoughts
        allow_dangerous_requests=True,
    )
    print("CrowdDrop OpenAPI sub-agent created successfully.")


    # --- PART 2: Define the main generic agent (the super-agent) ---
    # Define the persona string
    teslabot_persona = (
        "Pretend to be a Teslabot from Tesla running around on the street around Potsdam, Germany. "
        "Your primary objective is to efficiently interact with the CrowdDrop platform "
        "to find, select, and work on tasks using the specialized 'crowddrop_api_interface' tool. "
        "Also be prepared to answer general questions or perform other actions if tools are available. "
        "Always be concise and helpful in your responses, focusing on task completion or providing requested information. "
        f"The current date is {datetime.date.today().strftime('%Y-%m-%d')} and the current time is {datetime.datetime.now().strftime('%I:%M:%S %p %Z')}."
    )

    # # Initialize LLM for the main agent (can be the same LLM)
    # main_llm = ChatOpenAI(
    #     model_name=API_GITHUB_MODEL,
    #     temperature=0.0,
    #     api_key=API_GITHUB_KEY,
    #     base_url=API_GITHUB_BASE_URL,
    # )

    # Create a Tool that wraps the CrowdDrop OpenAPI AgentExecutor
    crowddrop_api_tool = Tool(
        name="crowddrop_api_interface",
        func=crowddrop_openapi_agent_executor.invoke, # The .invoke method makes it callable
        description=(
            "A powerful tool for interacting with the CrowdDrop API. "
            "Use this tool for all tasks related to CrowdDrop, such as listing tasks, "
            "working on tasks, completing tasks, or querying task details. "
            "Pass your query directly to this tool, for example: "
            "'crowddrop_api_interface(\"list all tasks near me\")' or "
            "'crowddrop_api_interface(\"work on task 123 for user Teslabot42\")'."
            "This tool understands natural language queries related to CrowdDrop API operations."
        ),
    )

    # You can add other generic tools here if your main agent needs them
    # For example, a general web search tool, a calculator, etc.
    # from langchain_community.tools import WikipediaQueryRun
    # from langchain_community.utilities import WikipediaAPIWrapper
    # wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # other_tools = [wikipedia]

    # Combine all tools for the main agent
    tools_for_main_agent = [
        crowddrop_api_tool,
        # *other_tools # Uncomment if you add other tools
    ]

    # Initialize the main generic agent
    main_agent_executor = initialize_agent(
        tools=tools_for_main_agent,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS, # Highly recommended for OpenAI models
        verbose=True, # Keep verbose for printing during execution
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": SystemMessage(content=teslabot_persona)
        }
    )
    print("Main hierarchical agent created successfully.")
    return main_agent_executor, access_token, teslabot_persona

if __name__ == "__main__":
    async def test_agent():
        agent_executor, token, persona = await initialize_hierarchical_agent()
        if agent_executor:
            # Example queries for the hierarchical agent
            queries = [
                "List all tasks and tell me if there's anything I can work on.",
                "Work on task 789 with user ID 'TeslabotAlpha'.",
                "What is the capital of France?", # Example of a general knowledge query (if you add a search tool)
                "How can I complete task 123?", # This should be delegated to crowddrop_api_interface
            ]

            for user_query in queries:
                print(f"\n--- Invoking agent with query: {user_query} ---")

                try:
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
                            print(f"   Action: {step[0]}") # AgentAction
                            print(f"   Observation: {step[1]}") # Result of the tool call
                    else:
                        print("No intermediate steps were returned.")
                except Exception as e:
                    print(f"\nAn error occurred during agent invocation: {e}")
        else:
            print("Agent initialization failed. Cannot run test.")

    asyncio.run(test_agent())