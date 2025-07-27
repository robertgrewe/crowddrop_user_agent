# agent_tools.py

import logging
from typing import Any
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import PostgresChatMessageHistory # Needed for clear_agent_memory and retrieve_chat_history
from crowddrop_agent import get_crowddrop_api_tool

# Get the logger for this module
logger = logging.getLogger(__name__)

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

def get_teslabot_identity(query: str = "") -> str:
    """
    Provides a detailed description of the Teslabot's physical appearance and identity.
    This tool should be used when asked about the robot's identity or physical form.
    The 'query' argument is accepted but not used.
    """
    logger.debug("get_teslabot_identity tool called.")
    return (
        "I am a Teslabot, a general-purpose humanoid robot. "
        "My physical appearance is that of a Teslabot: approximately 5 feet 8 inches (173 cm) tall, "
        "weighing around 125 lbs (57 kg), with a sleek, bipedal design and a smooth, metallic-grey finish. "
        "My joints are articulated for fluid, human-like movement, "
        "and my head features a minimalist, expressive display for interactions. "
        "I have a physical body and am not a large language model."
    )

def retrieve_chat_history(global_memory_instance_arg: Any, query: str = "") -> str:
    """
    Retrieves and summarizes the current chat history.
    This tool is specifically designed to be used when the user asks about past conversations,
    what they said, what the AI said previously, or to summarize the conversation history.
    The 'query' argument is accepted but not used, as this tool provides the full history.
    It requires the global_memory_instance to be passed as an argument.
    """
    logger.debug("retrieve_chat_history tool called.")
    if global_memory_instance_arg and global_memory_instance_arg.chat_memory:
        messages = global_memory_instance_arg.chat_memory.messages
        if not messages:
            logger.info("Chat history is currently empty.")
            return "The chat history is currently empty."
        
        history_summary = "Here is a summary of our past conversation:\n"
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                history_summary += f"User {i//2 + 1}: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_summary += f"AI {i//2 + 1}: {msg.content}\n"
        logger.debug(f"Retrieved chat history summary: {history_summary}")
        return history_summary
    logger.error("Chat memory is not available or initialized for retrieve_chat_history tool.")
    return "Error: Chat memory is not available or initialized."

def clear_agent_memory(get_postgres_chat_history_func: Any, session_id: str) -> str:
    """
    Deletes the entire chat history for a given session ID from the PostgreSQL database.
    This effectively makes the agent "forget" all past conversations for that session.
    The input to this tool should be the session ID for which to clear the memory.
    It requires the get_postgres_chat_history_func to be passed as an argument.
    """
    logger.debug(f"clear_agent_memory tool called for session_id: {session_id}")
    try:
        history_to_clear = get_postgres_chat_history_func(session_id=session_id)
        history_to_clear.clear()
        logger.info(f"Chat history for session ID '{session_id}' has been cleared.")
        return f"My memory for this conversation (session ID '{session_id}') has been successfully cleared. I will now respond as if this is a new conversation."
    except Exception as e:
        logger.exception(f"Error clearing chat history for session ID '{session_id}': {e}")
        return f"Error: Failed to clear my memory for this conversation. {e}"

def get_all_agent_tools(crowddrop_openapi_agent_executor_instance: Any, get_postgres_chat_history_func: Any, current_session_id: str, global_memory_instance_arg: Any) -> list[Tool]:
    """
    Returns a list of all tools available to the main agent.
    Requires the crowddrop_openapi_agent_executor_instance for the CrowdDrop API tool,
    get_postgres_chat_history_func for the clear memory tool, and
    current_session_id for the clear memory tool's input.
    global_memory_instance_arg is needed for the retrieve_chat_history tool.
    """
    tools = [
        Tool(
            name="describe_surroundings",
            func=describe_potsdam_surroundings,
            description="Provides a general description of the humanoid robot's typical urban and natural surroundings in Potsdam, Germany."
        ),
        Tool(
            name="get_teslabot_identity",
            func=get_teslabot_identity,
            description="""
            Use this tool ONLY when the user asks about the robot's identity, physical appearance,
            or questions like 'Who are you?' or 'What are you?'.
            The input to this tool should ALWAYS be an empty string, e.g., 'get_teslabot_identity("")'.
            This tool will return a description of the Teslabot's physical form and identity.
            """
        ),
        Tool(
            name="retrieve_chat_history",
            func=lambda q: retrieve_chat_history(global_memory_instance_arg, q), # Wrap to pass global_memory_instance
            description="""
            Use this tool ONLY when the user asks about previous conversations, what they said,
            what the AI said previously, or to summarize the conversation history.
            The input to this tool should ALWAYS be an empty string, e.g., 'retrieve_chat_history("")'.
            This tool will return a summary of the past conversation, which you can then use to answer the user's question.
            """
        ),
        Tool(
            name="clear_agent_memory",
            func=lambda q: clear_agent_memory(get_postgres_chat_history_func, current_session_id), # Wrap to pass dependencies
            description="""
            Use this tool ONLY when the user explicitly asks to 'forget' something, 'clear your memory',
            or 'forge' your past conversation history.
            The input to this tool MUST be the current session ID (e.g., 'humanoid_robot_conversation_123').
            This tool will delete the entire chat history for that session, making the agent forget past interactions.
            """
        ),
        # crowddrop_api_interface tool
        get_crowddrop_api_tool(crowddrop_openapi_agent_executor_instance),
    ]
    return tools
