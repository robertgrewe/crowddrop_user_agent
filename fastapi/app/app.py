from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import json  # Import the json module

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

class ChatRequest(BaseModel):
    model: str = "llama3"  # Specify the Ollama model you want to use
    prompt: str

@app.post("/generate/ollama")
async def generate_ollama(request: ChatRequest):
    try:
        ollama_base_url = "http://ollama:11434/api"  # Use the service name 'ollama'
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

# Run the server with the command:
# uvicorn main:app --reload
# (assuming your FastAPI code is in main.py)