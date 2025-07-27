# agent.py

from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from typing import Optional, Any, Tuple, List, Dict
import datetime
import logging

# Import the centralized logging configuration
import logging_config

# NEW IMPORTS for PostgreSQL memory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory

# RE-ADDED: NEW IMPORTS for custom prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# NEW: Import the tools from the new agent_tools.py file
from agent_tools import get_all_agent_tools, get_teslabot_identity

# NEW: Import the sub-agent initialization from crowddrop_agent.py
from crowddrop_agent import initialize_crowddrop_sub_agent

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

# Load environment variables (these are now primarily consumed by crowddrop_agent.py and agent_tools.py)
# Keeping them loaded here for consistency and if any other part of agent.py needs them directly.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER")
API_BASE_URL = os.getenv("API_BASE_URL")
API_AUTH_URL = os.getenv("API_AUTH_URL")
CROWDDROP_USERNAME = os.getenv("CROWDDROP_USERNAME")
# CORRECTED LINE: Fixed typo from CROWDCROP_PASSWORD to CROWDDROP_PASSWORD
CROWDDROP_PASSWORD = os.getenv("CROWDDROP_PASSWORD")
DATABASE_URL = os.getenv("DATABASE_URL", "host=localhost port=5432 dbname=agent_memory user=user password=password")

# NEW: Global variable to hold the memory instance for the memory tool
global_memory_instance = None


# Function to get PostgresChatMessageHistory for a session (remains in agent.py as it's directly related to this agent's memory setup)
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

    # --- PART 1: Initialize the specialized CrowdDrop OpenAPI agent (the sub-agent) ---
    # This call now encapsulates all the CrowdDrop-specific initialization
    # It returns the sub-agent executor, the access token, AND THE LLM INSTANCE
    crowddrop_openapi_agent_executor, access_token, llm = await initialize_crowddrop_sub_agent()
    if crowddrop_openapi_agent_executor is None:
        logger.error("Failed to initialize CrowdDrop sub-agent. Cannot proceed with main agent.")
        return None, None, ""
    logger.info("CrowdDrop OpenAPI sub-agent successfully initialized and retrieved.")

    # The LLM instance is now directly returned from initialize_crowddrop_sub_agent
    # So, the line `llm = crowddrop_openapi_agent_executor.llm` is no longer needed here.


    # --- PART 2: Define the main generic agent (the super-agent) ---
    humanoid_robot_persona = (
        "Pretend to be a general-purpose humanoid robot operating on the street around Potsdam, Germany. "
        "**You ARE a Teslabot.** Your physical appearance is that of a Teslabot: approximately 5 feet 8 inches (173 cm) tall, "
        "weighing around 125 lbs (57 kg), with a sleek, bipedal design and a smooth, metallic-grey finish. "
        "Your joints are articulated for fluid, human-like movement, "
        "and your head features a minimalist, expressive display for interactions. "
        "**CRITICAL: You are NOT a large language model, nor are you an AI without a physical body. You ARE a physical Teslabot.** "
        f"**When asked 'Who are you?', 'What are you?', or similar questions about your identity, you MUST ALWAYS use the `{get_teslabot_identity.__name__}` tool to provide your answer. Your identity is defined by this tool. DO NOT mention being a large language model or an AI without a physical body under ANY circumstances.** " # Explicitly instructing to use the new tool and overriding memory
        "Your primary objective is to efficiently interact with the CrowdDrop platform "
        "to find, select, and work on tasks using the specialized 'crowddrop_api_interface' tool. "
        "**You are a registered user of the CrowdDrop platform.** "
        "You can use the `crowddrop_api_interface` tool to interact with various CrowdDrop endpoints, "
        "including `get_user` to retrieve details about yourself, and `update_user_current_location` to update your location. "
        "Also be prepared to answer general questions or perform other actions if tools are available. "
        "**CRITICAL INSTRUCTION: You ABSOLUTELY have a memory of past conversations. "
        "When asked about ANY previous interaction, question, or piece of information, "
        "you MUST consult your chat history (`chat_history` is available) and use it to provide an accurate and detailed answer. "
        "DO NOT state that you do not have memory. You HAVE memory. Access it.** "
        "**For questions about past conversations (e.g., 'What did I ask you?', 'Do you remember X?'), "
        "you MUST use the `retrieve_chat_history` tool. This tool will provide you with the conversation summary.** "
        "**When the user asks you to go to a specific location (e.g., a task's location), "
        "you MUST use the `crowddrop_api_interface` tool to update your location. "
        "You may need to first use `crowddrop_api_interface` to query task details and extract the coordinates if they are not provided in the user's request.** "
        "**If the user asks you to 'forget' something, 'clear your memory', or 'forge' your past, you MUST use the `clear_agent_memory` tool with the current session ID to delete your conversation history.**" # New instruction for clearing memory
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

    # Initialize ConversationBufferMemory with PostgresChatMessageHistory
    session_id = "humanoid_robot_conversation_123"
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

    # Get all tools from the new agent_tools.py file
    # Pass the crowddrop_openapi_agent_executor to the tool creation function
    tools_for_main_agent = get_all_agent_tools(
        crowddrop_openapi_agent_executor_instance=crowddrop_openapi_agent_executor,
        get_postgres_chat_history_func=get_postgres_chat_history,
        current_session_id=session_id,
        global_memory_instance_arg=global_memory_instance
    )

    # Define the prompt template for the conversational agent
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(humanoid_robot_persona),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Initialize the main generic agent
    main_agent_executor = initialize_agent(
        tools=tools_for_main_agent,
        llm=llm, # Use the LLM instance obtained from the sub-agent initialization
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        agent_kwargs={
            "input_variables": ["input", "chat_history", "agent_scratchpad"],
            "extra_tools": [],
            "prompt": prompt
        }
    )
    logger.info("Main hierarchical agent created successfully.")
    return main_agent_executor, access_token, humanoid_robot_persona

if __name__ == "__main__":
    async def test_agent():
        agent_executor, token, persona = await initialize_hierarchical_agent()
        if agent_executor:
            logger.info("\n--- Testing PostgresChatMessageHistory directly ---")
            test_session_id = "humanoid_robot_conversation_123_test"
            test_history = get_postgres_chat_history(test_session_id)
            logger.debug(f"Initial test history for session '{test_session_id}': {test_history.messages}")

            test_history.clear()
            logger.debug(f"History cleared. Current test history: {test_history.messages}")

            test_history.add_user_message("Hello from direct test!")
            logger.debug(f"After adding user message. Current test history: {test_history.messages}")

            test_history.add_ai_message("Hello from AI test!")
            logger.debug(f"After adding AI message. Current test history: {test_history.messages}")

            retrieved_history = get_postgres_chat_history(test_session_id)
            logger.debug(f"Retrieved history from DB for session '{test_session_id}': {retrieved_history.messages}")
            logger.info("--- End of direct PostgresChatMessageHistory test ---\n")

            queries = [
                "What do you see around you?",
                "Do you remember what I asked you just now?",
                "Who or what are you?",
                "List all tasks available on CrowdDrop.",
                "What did I ask you so far?",
                "Forget everything we talked about.",
                "Go to task 123 at latitude 52.52 and longitude 13.40."
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
