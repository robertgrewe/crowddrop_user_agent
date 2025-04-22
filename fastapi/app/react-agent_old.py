import os
import random

from langchain import agents
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI
from openai import AzureOpenAI

import pytz
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# token = os.environ["API_GITHUB_KEY"]
# endpoint = "https://models.inference.ai.azure.com"
# model_name = "gpt-4o-mini"

# llm = ChatOpenAI(
#     model=model_name,
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=token,
#     base_url=endpoint
# )
    
# cloud model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version=os.getenv("AZURE_VERSION"),  # or your api version
    azure_endpoint=os.getenv("AZURE_BASE_URL"),  # or your endpoint
    openai_api_version="2024-06-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# local model running in docker container
llm = OllamaLLM(
    model="llama3", 
    temperature=0.0, 
    max_tokens=None, 
    timeout=None, 
    max_retries=2, 
    base_url="http://localhost:11434",
    verbose=True)

@tool
def get_current_username(input: str) -> str:
    "Get the username of the current user."
    return "Dennis"

@tool
def get_current_location(username: str) -> str:
    "Get the current timezone location of the user for a given username."
    print(username)
    if "Dennis" in username:
        return "Europe/Berlin"
    else:
        return "America/New_York"

@tool
def get_current_time(location: str) -> str:
    "Get the current time in the given location. The pytz is used to get the timezone for that location. Location names should be in a format like America/Seattle, Asia/Bangkok, Europe/London. Anything in Germany should be Europe/Berlin"
    try:
        print("get current time for location: ", location)
        location = str.replace(location, " ", "")
        location = str.replace(location, "\"", "")
        location = str.replace(location, "\n", "")
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except Exception as e:
        print("Error: ", e)
        return "Sorry, I couldn't find the timezone for that location."

@tool  
def get_current_weather(location: str) -> str:  
    "Mock the current weather for a given location with random weather conditions."  
    weather_conditions = ['sunny', 'rainy', 'snowy']  
    # Randomly choose a weather condition  
    current_condition = random.choice(weather_conditions)  
      
    # Create a response based on the chosen condition  
    if current_condition == 'sunny':  
        return f"The current weather in {location} is sunny."  
    elif current_condition == 'rainy':  
        return f"The current weather in {location} is rainy."  
    elif current_condition == 'snowy':  
        return f"The current weather in {location} is snowy."  
    else:  
        return "Sorry, I couldn't determine the weather for that location."

@tool  
def get_appropriate_item_for_weather(weather: str) -> str:  
    "Get an appropriate item based on the current weather condition."  
    if 'sunny' in weather.lower():  
        return "It's sunny outside. Don't forget to grab a hat!"  
    elif 'rainy' in weather.lower():  
        return "It's rainy outside. Make sure to take an umbrella!"  
    elif 'snowy' in weather.lower():  
        return "It's snowy outside. You might need a coat and snow boots!"  
    else:  
        return "The weather is pleasant. No special items needed."  
  

tools = []
tools = [get_current_time]
tools = [get_current_username, get_current_location, get_current_time, get_current_weather, get_appropriate_item_for_weather]

commandprompt = '''
    ##
    You are a helpfull assistent and should respond to user questions.
    If you cannot answer a question then say so explicitly and stop.
    
    '''

promptString = commandprompt +  """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]. Make sure that Actions are not commands. They should be the name of the tool to use.

Action Input: the input to the action according to the tool signature

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}

"""
prompt = PromptTemplate.from_template(promptString)

agent = create_react_agent(llm, tools, prompt)

agent_executor = agents.AgentExecutor(
        name="Tools Agent",
        agent=agent, tools=tools,  verbose=True, handle_parsing_errors=True, max_iterations=10, return_intermediate_steps=True,
    )

#input = "What is the current time here?"
#input = "Do I need an umbrella today?"
#input = "I am in Berlin. Do I need an umbrella?"
#input = "I am Rob. Do I need an umbrella?"
input = "What do I need today?"

response = agent_executor.invoke(
    {"input": input},
)
       