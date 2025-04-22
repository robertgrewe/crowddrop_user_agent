# Docker

## build docker
docker build -f docker/Dockerfile -t fastapi:latest .

## run docker containter
docker run -p 8000:8000 fastapi:latest

## open Swagger
http://127.0.0.1:8000/docs

# Resources

## Azure ChatGPT Documentation
https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python-new

# Old

# used python version
Python 3.12.0

# required packages 
py -m pip install openai
py -m pip install uvicorn
py -m pip install fastapi
py -m pip install python-dotenv

# start a simple test script
py main.py

# store api key in .env
AZURE_OPENAI_API_KEY=ADD_API_KEY_HERE

# start FastAPI
py -m uvicorn myAPI:app --reload  

## open Swagger
http://127.0.0.1:8000/docs