# agent.py

from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.utilities import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool, StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Import HumanMessage, AIMessage for direct memory testing
from dotenv import load_dotenv, find_dotenv
import json
import os
import requests
import urllib.parse
import subprocess
import asyncio
from typing import Optional, Any, Tuple, List, Dict
import datetime

# NEW IMPORTS for PostgreSQL memory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory

# RE-ADDED: NEW IMPORTS for custom prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate


load_dotenv(find_dotenv())

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
            print(f"Authentication request failed (curl): {stderr.decode()}")
            return None

        response_data = json.loads(stdout.decode())
        print(f"response_data: {response_data}")
        return response_data.get("id_token")

    except Exception as e:
        print(f"Authentication request failed (exception): {e}")
        return None

def describe_potsdam_surroundings(query: str) -> str:
    """
    Describes the general surroundings of a Teslabot in Potsdam, Germany.
    This is a static description.
    The 'query' argument is accepted but not used, as this tool provides a fixed description.
    """
    return (
        "As a Teslabot operating in Potsdam, Germany, I observe a mix of historic architecture, "
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
    global global_memory_instance
    if global_memory_instance and global_memory_instance.chat_memory:
        messages = global_memory_instance.chat_memory.messages
        if not messages:
            return "The chat history is currently empty."
        
        history_summary = "Here is a summary of our past conversation:\n"
        for i, msg in enumerate(messages):
            # Only include the content of the message, not metadata
            if isinstance(msg, HumanMessage):
                history_summary += f"User {i//2 + 1}: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_summary += f"AI {i//2 + 1}: {msg.content}\n"
        return history_summary
    return "Error: Chat memory is not available or initialized."

# NEW: Function to get PostgresChatMessageHistory for a session
def get_postgres_chat_history(session_id: str) -> PostgresChatMessageHistory:
    """
    Returns a PostgresChatMessageHistory instance for a given session ID.
    """
    print(f"Connecting to PostgreSQL for session_id: {session_id} using DATABASE_URL: {DATABASE_URL}")
    return PostgresChatMessageHistory(
        connection_string=DATABASE_URL,
        session_id=session_id
    )

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
    # Corrected API_OPENAPI_SPEC_URL to use API_BASE_URL for local spec
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
    # UPDATED: Added instructions about memory and emphasized it
    teslabot_persona = (
        "Pretend to be a humanoid Teslabot from Tesla running around on the street around Potsdam, Germany. "
        "Your primary objective is to efficiently interact with the CrowdDrop platform "
        "to find, select, and work on tasks using the specialized 'crowddrop_api_interface' tool. "
        "Also be prepared to answer general questions or perform other actions if tools are available. "
        "**CRITICAL INSTRUCTION: You ABSOLUTELY have a memory of past conversations. "
        "When asked about ANY previous interaction, question, or piece of information, "
        "you MUST consult your chat history (`chat_history` is available) and use it to provide an accurate and detailed answer. "
        "DO NOT state that you do not have memory. You HAVE memory. Access it.** "
        "**For questions about past conversations (e.g., 'What did I ask you?', 'Do you remember X?'), "
        "you MUST use the `retrieve_chat_history` tool. This tool will provide you with the conversation summary.** "
        "Always be concise and helpful in your responses, focusing on task completion or providing requested information. "
        f"The current date is {datetime.date.today().strftime('%Y-%m-%d')} and the current time is {datetime.datetime.now().strftime('%I:%M:%S %p %Z')}."
        "\n\n" # Add a newline to separate persona from format instructions
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
    print("Teslabot persona defined successfully.")

    # Create the main agent's tools

    # Create the describe surroundings tool
    surroundings_tool = Tool(
        name="describe_surroundings",
        func=describe_potsdam_surroundings,
        description="Provides a general description of the Teslabot's typical urban and natural surroundings in Potsdam, Germany."
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
        surroundings_tool,
        crowddrop_api_tool,
        memory_retrieval_tool, # Add the new memory retrieval tool
        # *other_tools # Uncomment if you add other tools
    ]

    # NEW: Initialize ConversationBufferMemory with PostgresChatMessageHistory
    # For demonstration, we'll use a fixed session ID. In a real application,
    # this would come from the user's request (e.g., a user ID or conversation ID).
    session_id = "teslabot_conversation_123" # Replace with a dynamic session ID in app.py
    memory = ConversationBufferMemory(
        chat_memory=get_postgres_chat_history(session_id=session_id),
        return_messages=True,
        memory_key="chat_history", # This is the key that the agent will look for in the prompt
        input_key="input" # Assuming your agent's input key is "input"
    )
    # Set the global memory instance for the memory retrieval tool
    global global_memory_instance
    global_memory_instance = memory

    print(f"Agent memory initialized with PostgreSQL for session: {session_id}")

    # RE-ADDED: Define the prompt template for the conversational agent
    # This explicitly includes the chat history in the prompt
    # This is crucial for the LLM to see the memory
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(teslabot_persona),
            MessagesPlaceholder(variable_name="chat_history"), # This will inject the memory
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize the main generic agent
    main_agent_executor = initialize_agent(
        tools=tools_for_main_agent,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, # Keep verbose for printing during execution
        handle_parsing_errors=True, # Keep this to allow agent to retry on parsing errors
        memory=memory, # <--- PASS THE MEMORY OBJECT HERE
        # Pass the custom prompt directly to the agent's kwargs
        # This is the most reliable way to ensure the prompt is used
        agent_kwargs={
            "input_variables": ["input", "chat_history", "agent_scratchpad"], # Explicitly define input variables
            "extra_tools": [], # If you had tools specific to the agent, not the executor
            "prompt": prompt # Pass the constructed prompt here
        }
    )
    print("Main hierarchical agent created successfully.")
    return main_agent_executor, access_token, teslabot_persona

if __name__ == "__main__":
    async def test_agent():
        agent_executor, token, persona = await initialize_hierarchical_agent()
        if agent_executor:
            # DEBUGGING: Explicitly test PostgresChatMessageHistory
            print("\n--- Testing PostgresChatMessageHistory directly ---")
            test_session_id = "teslabot_conversation_123_test"
            test_history = get_postgres_chat_history(test_session_id)
            print(f"Initial test history for session '{test_session_id}': {test_history.messages}")

            # Clear any existing history for this test session
            test_history.clear()
            print(f"History cleared. Current test history: {test_history.messages}")

            # Add a message
            test_history.add_user_message("Hello from direct test!")
            print(f"After adding user message. Current test history: {test_history.messages}")

            # Add an AI message
            test_history.add_ai_message("Hello from AI test!")
            print(f"After adding AI message. Current test history: {test_history.messages}")

            # Retrieve messages again to confirm persistence
            retrieved_history = get_postgres_chat_history(test_session_id)
            print(f"Retrieved history from DB for session '{test_session_id}': {retrieved_history.messages}")
            print("--- End of direct PostgresChatMessageHistory test ---\n")


            # Example queries for the hierarchical agent
            queries = [
                "What do you see around you?",
                "Do you remember what I asked you just now?", # Test memory
                "Who or what are you?", # Test memory and persona
                "List all tasks available on CrowdDrop.", # Example CrowdDrop API call
                "What did I ask you so far?" # Test the new memory tool
            ]

            for user_query in queries:
                print(f"\n--- Invoking agent with query: {user_query} ---")
                # DEBUGGING: Print the chat history before invocation
                # This `memory` object is the one used by the agent_executor
                print(f"Current Chat History (from agent's memory before invoke): {memory.chat_memory.messages}")

                try:
                    response = await agent_executor.ainvoke(
                        {"input": user_query},
                        return_intermediate_steps=False
                    )

                    print("\n--- Agent Response ---")
                    print(f"Final Answer: {response.get('output')}")

                    # DEBUGGING: Print the chat history after invocation to see if it was updated
                    print(f"Current Chat History (from agent's memory after invoke): {memory.chat_memory.messages}")

                except Exception as e:
                    print(f"\nAn error occurred during agent invocation: {e}")
        else:
            print("Agent initialization failed. Cannot run test.")

    asyncio.run(test_agent())
