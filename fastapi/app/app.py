from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import json
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequestRaw(BaseModel):
    model: str = "llama3"
    prompt: str

class Message(BaseModel):
    role: str
    content: str

    def to_langchain_message(self) -> BaseMessage:
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        else:
            raise ValueError(f"Invalid role: {self.role}")

class ChatRequestLangchain(BaseModel):
    model: str = "llama3"
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float | None = None

@app.post("/generate/ollama_raw")
async def generate_ollama_raw(request: ChatRequestRaw):
    try:
        ollama_base_url = "http://ollama:11434/api"
        ollama_generate_url = f"{ollama_base_url}/generate"

        payload = {
            "model": request.model,
            "prompt": request.prompt
        }

        response = requests.post(ollama_generate_url, json=payload, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = line.decode('utf-8')
                    data = json.loads(json_data)
                    if 'response' in data:
                        full_response += data['response']
                    if 'error' in data:
                        raise HTTPException(status_code=500, detail=data['error'])
                    if 'done' in data and data['done']:
                        break
                except json.JSONDecodeError:
                    print(f"Could not decode line: {line}")
                    continue

        return {"response": full_response}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/ollama_langchain")
async def chat_ollama_langchain(request: ChatRequestLangchain):
    try:
        llm = Ollama(
            model=request.model,
            temperature=request.temperature,
            base_url="http://ollama:11434",
            verbose=True
        )

        langchain_messages = [msg.to_langchain_message() for msg in request.messages]

        config = {}
        if request.max_tokens is not None:
            config["max_tokens"] = request.max_tokens
        if request.timeout is not None:
            config["timeout"] = request.timeout

        response = llm.invoke(langchain_messages, config=RunnableConfig(**config))
        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server with the command:
# uvicorn main:app --reload
# (assuming your FastAPI code is in main.py)