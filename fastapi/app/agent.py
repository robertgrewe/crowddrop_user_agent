# agent.py

from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Added HumanMessage, AIMessage for memory tool
from dotenv import load_dotenv, find_dotenv
import json
import os
import requests
import urllib.parse
import subprocess
import asyncio
from typing import Optional, Any, Tuple, List, Dict
import datetime
import logging # Import the logging module

# Import the centralized logging configuration
import logging_config # This will automatically run configure_logging()

# NEW IMPORTS for PostgreSQL memory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory

# RE-ADDED: NEW IMPORTS for custom prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate


load_dotenv(find_dotenv())

# Get the logger for this module (it will inherit configuration from logging_config.py)
logger = logging.getLogger(__name__)


# Load authentication details from environment variables (from .env)

# Get the model provider from the environment variable
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")

# CrowdDrop API
API_BASE_URL = os.getenv("API_BASE_URL")
API_AUTH_URL = os.getenv("API_AUTH_URL")
API_OPENAPI_SPEC_URL = os.getenv("API_BASE_URL") + "/openapi.json" # Ensure this points to your local FastAPI OpenAPI spec
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME")
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD")

# NEW: Database URL for PostgreSQL memory
DATABASE_URL = os.getenv("DATABASE_URL", "host=localhost port=5432 dbname=agent_memory user=user password=password")

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

def describe_potsdam_surroundings(query: str) -> str:
    """
    Describes the general surroundings of a humanoid robot in Potsdam, Germany.
    This is a static description.
    The 'query' argument is accepted but not used, as this tool provides a fixed description.
    """
    logger.debug("describe_potsdam_surroundings tool called.")
    return (
        "As a humanoid robot operating in Potsdam, Germany, I observe a mix of historic architecture, "
        "including palaces and old townhouses, alongside more modern urban structures. "
        "There are numerous well-maintained parks and green spaces, such as Sanssouci Park, "
        "and the city is dotted with beautiful lakes and canals connected to the Havel River. "
        "I typically see a combination of paved roads, sidewalks, trees, residential buildings, "
        "shops, and other people going about their day. Depending on the exact location, "
        "there might be specific landmarks, cafes, or public transport stops."
    )

# NEW: Global variable to hold the memory instance for the memory tool
global_memory_instance = None

def retrieve_chat_history(query: str = "") -> str:
    """
    Retrieves and summarizes the current chat history.
    This tool is specifically designed to be used when the user asks about past conversations,
    what they said, what the AI said previously, or to summarize the conversation history.
    The 'query' argument is accepted but not used, as this tool provides the full history.
    """
    logger.debug("retrieve_chat_history tool called.")
    global global_memory_instance
    if global_memory_instance and global_memory_instance.chat_memory:
        messages = global_memory_instance.chat_memory.messages
        if not messages:
            logger.info("Chat history is currently empty.")
            return "The chat history is currently empty."
        
        history_summary = "Here is a summary of our past conversation:\n"
        for i, msg in enumerate(messages):
            # Only include the content of the message, not metadata
            if isinstance(msg, HumanMessage):
                history_summary += f"User {i//2 + 1}: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_summary += f"AI {i//2 + 1}: {msg.content}\n"
        logger.debug(f"Retrieved chat history summary: {history_summary}")
        return history_summary
    logger.error("Chat memory is not available or initialized for retrieve_chat_history tool.")
    return "Error: Chat memory is not available or initialized."

# NEW: Function to get PostgresChatMessageHistory for a session
def get_postgres_chat_history(session_id: str) -> PostgresChatMessageHistory:
    """
    Returns a PostgresChatMessageHistory instance for a given session ID.
    """
    logger.info(f"Connecting to PostgreSQL for session_id: {session_id} using DATABASE_URL: {DATABASE_URL}")
    return PostgresChatMessageHistory(
        connection_string=DATABASE_URL,
        session_id=session_id
    )

async def initialize_hierarchical_agent() -> Tuple[Any, Optional[str], str]:
    """
    Initializes and returns the LangChain hierarchical agent, the access token,
    and the humanoid robot persona string.
    """
    logger.info("Initializing hierarchical agent within agent.py...")

    # Authenticate and get token
    access_token = authenticate_and_get_token(CROWDDROP_USERNAME, CROWDDROP_PASSWORD)
    if access_token is None:
        logger.error("Authentication failed. Cannot initialize agent without token.")
        return None, None, ""

    # --- PART 1: Initialize the specialized CrowdDrop OpenAPI agent (the sub-agent) ---
    logger.info("Initializing CrowdDrop OpenAPI sub-agent...")
    # Load and modify OpenAPI spec
    openapi_spec_url = API_BASE_URL + "/openapi.json"
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

        logger.info(f"Modified OpenAPI Spec Servers: {raw_openai_api_spec.get('servers')}")
        openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
        logger.info("Successfully loaded and modified OpenAPI spec from URL.")
    except requests.exceptions.RequestException as e:
        logger.exception(f"Error fetching OpenAPI spec from URL: {e}")
        return None, None, "" # Return None if spec loading fails

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
        API_GEMINI_MODEL = os.getenv("API_GEMINI_MODEL", "gemini-1.5-flash") # Changed back to a stable model

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

    # Configure RequestsWrapper with the Authorization header for the OpenAPI agent
    openapi_requests_wrapper = RequestsWrapper(headers={"Authorization": f"Bearer {access_token}", "accept": "application/json"})

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


    # --- PART 2: Define the main generic agent (the super-agent) ---
    # Define the persona string - Updated to be a generic humanoid robot
    humanoid_robot_persona = (
        "Pretend to be a general-purpose humanoid robot operating on the street around Potsdam, Germany. "
        "Your primary objective is to efficiently interact with the CrowdDrop platform "
        "to find, select, and work on tasks using the specialized 'crowddrop_api_interface' tool. "
        "Also be prepared to answer general questions or perform other actions if tools are available. "
        "**CRITICAL INSTRUCTION: You ABSOLUTELY have a memory of past conversations. "
        "When asked about ANY previous interaction, question, or piece of information, "
        "you MUST consult your chat history (`chat_history` is available) and use it to provide an accurate and detailed answer. "
        "DO NOT state that you do not have memory. You HAVE memory. Access it.** "
        "**For questions about past conversations (e.g., 'What did I ask you?', 'Do you remember X?'), "
        "you MUST use the `retrieve_chat_history` tool. This tool will provide you with the conversation summary.** "
        "**You are a user of the CrowdDrop platform and can update your own location by calling the appropriate endpoint "
        "via the `crowddrop_api_interface` tool. When the user asks you to go to a specific location (e.g., a task's location), "
        "you MUST use the `crowddrop_api_interface` tool to update your location. "
        "You may need to first use `crowddrop_api_interface` to query task details and extract the coordinates if they are not provided in the user's request.** "
        "Always be concise and helpful in your responses, focusing on task completion or providing requested information. "
        f"The current date is {datetime.date.today().strftime('%Y-%m-%d')} and the current time is {datetime.datetime.now().strftime('%I:%M:%S %p %Z')}."
        "\n\n"
        "**IMPORTANT: You MUST strictly adhere to the following output format for ALL responses.**\n"
        "If you need to think, use 'Thought:'. If you need to use a tool, use 'Action:' and 'Action Input:'. "
        "If you have the final answer, use 'Final Answer:'.\n"
        "Do NOT include any other conversational text outside of these structured tags, especially when providing a final answer or tool output.\n"
        "Use the following format:\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: the final answer to the original input question"
    )
    logger.info("Humanoid robot persona defined successfully.")

    # Create the main agent's tools

    # Create the describe surroundings tool - Updated description
    surroundings_tool = Tool(
        name="describe_surroundings",
        func=describe_potsdam_surroundings,
        description="Provides a general description of the humanoid robot's typical urban and natural surroundings in Potsdam, Germany."
    )

    # Create the memory retrieval tool
    memory_retrieval_tool = Tool(
        name="retrieve_chat_history",
        func=retrieve_chat_history,
        description="""
        Use this tool ONLY when the user asks about previous conversations, what they said,
        what the AI said previously, or to summarize the conversation history.
        The input to this tool should ALWAYS be an empty string, e.g., 'retrieve_chat_history("")'.
        This tool will return a summary of the past conversation, which you can then use to answer the user's question.
        """
    )

    # Create a Tool that wraps the CrowdDrop OpenAPI AgentExecutor
    crowddrop_api_tool = Tool(
        name="crowddrop_api_interface",
        func=crowddrop_openapi_agent_executor.invoke, # The .invoke method makes it callable
        description=(
            "A powerful tool for interacting with the CrowdDrop API. "
            "Use this tool for all tasks related to CrowdDrop, such as listing tasks, "
            "working on tasks, completing tasks, querying task details (including coordinates), "
            "or **updating the humanoid robot's own location**. " # Updated description
            "Pass your query directly to this tool, for example: "
            "'crowddrop_api_interface(\"list all tasks near me\")', "
            "'crowddrop_api_interface(\"query details for task 123\")', or "
            "'crowddrop_api_interface(\"update my location to latitude 52.123 and longitude 13.456\")'."
            "This tool understands natural language queries related to CrowdDrop API operations."
        ),
    )

    # Combine all tools for the main agent
    tools_for_main_agent = [
        surroundings_tool,
        crowddrop_api_tool,
        memory_retrieval_tool,
    ]

    # Initialize ConversationBufferMemory with PostgresChatMessageHistory
    session_id = "humanoid_robot_conversation_123" # Updated session ID for clarity
    memory = ConversationBufferMemory(
        chat_memory=get_postgres_chat_history(session_id=session_id),
        return_messages=True,
        memory_key="chat_history",
        input_key="input"
    )
    # Set the global memory instance for the memory retrieval tool
    global global_memory_instance
    global_memory_instance = memory

    logger.info(f"Agent memory initialized with PostgreSQL for session: {session_id}")

    # Define the prompt template for the conversational agent
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(humanoid_robot_persona), # Updated persona variable
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize the main generic agent
    main_agent_executor = initialize_agent(
        tools=tools_for_main_agent,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, # Using CONVERSATIONAL_REACT_DESCRIPTION with custom prompt
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        agent_kwargs={
            "input_variables": ["input", "chat_history", "agent_scratchpad"],
            "extra_tools": [],
            "prompt": prompt # Pass the constructed prompt here
        }
    )
    logger.info("Main hierarchical agent created successfully.")
    return main_agent_executor, access_token, humanoid_robot_persona # Updated return value

if __name__ == "__main__":
    async def test_agent():
        agent_executor, token, persona = await initialize_hierarchical_agent()
        if agent_executor:
            # DEBUGGING: Explicitly test PostgresChatMessageHistory
            logger.info("\n--- Testing PostgresChatMessageHistory directly ---")
            test_session_id = "humanoid_robot_conversation_123_test" # Updated test session ID
            test_history = get_postgres_chat_history(test_session_id)
            logger.debug(f"Initial test history for session '{test_session_id}': {test_history.messages}")

            # Clear any existing history for this test session
            test_history.clear()
            logger.debug(f"History cleared. Current test history: {test_history.messages}")

            # Add a message
            test_history.add_user_message("Hello from direct test!")
            logger.debug(f"After adding user message. Current test history: {test_history.messages}")

            # Add an AI message
            test_history.add_ai_message("Hello from AI test!")
            logger.debug(f"After adding AI message. Current test history: {test_history.messages}")

            # Retrieve messages again to confirm persistence
            retrieved_history = get_postgres_chat_history(test_session_id)
            logger.debug(f"Retrieved history from DB for session '{test_session_id}': {retrieved_history.messages}")
            logger.info("--- End of direct PostgresChatMessageHistory test ---\n")


            # Example queries for the hierarchical agent
            queries = [
                "What do you see around you?",
                "Do you remember what I asked you just now?", # Test memory
                "Who or what are you?", # Test memory and persona
                "List all tasks available on CrowdDrop.", # Example CrowdDrop API call
                "What did I ask you so far?", # Test the new memory tool
                "Go to task 123 at latitude 52.52 and longitude 13.40." # Test new location update via CrowdDrop API
            ]

            for user_query in queries:
                logger.info(f"\n--- Invoking agent with query: {user_query} ---")
                # This `memory` object is the one used by the agent_executor
                logger.debug(f"Current Chat History (from agent's memory before invoke): {memory.chat_memory.messages}")

                try:
                    response = await agent_executor.ainvoke(
                        {"input": user_query},
                        return_intermediate_steps=False
                    )

                    logger.info("\n--- Agent Response ---")
                    logger.info(f"Final Answer: {response.get('output')}")

                    logger.debug(f"Current Chat History (from agent's memory after invoke): {memory.chat_memory.messages}")

                except Exception as e:
                    logger.exception(f"\nAn error occurred during agent invocation: {e}")
        else:
            logger.error("Agent initialization failed. Cannot run test.")

    asyncio.run(test_agent())
