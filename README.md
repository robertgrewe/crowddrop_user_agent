# crowddrop_user_agent
This repo contains code for an AI Agent pretending to be a humanoid robot agent logged in as a user based on CrowdDrop.

# Docker

# start docker compose
docker compose up -d		# start containers
docker-compose up --build  	# build containers again

# stop docker compose
docker compose down

# docker container
docker ps -a					# list
docker stop <container_id>		# stop
docker rm <container_id>		# remove

# docker images
docker rmi <image_id>			# remove

# Ollama Docker Image
https://hub.docker.com/r/ollama/ollama

## Open Web UI

To connect the AI agent to OpenWebUi use the following details:

```
http://host.docker.internal:8003/v1
```

To talk to ollama model use:

```
http://host.docker.internal:11434
```

## Models
Download Ollama models by running:

```docker exec -it ollama ollama run llama3``` # llama3:latest model on Ollama does not have built-in support for tool calling (also known as function calling)

```docker exec -it ollama ollama run llama3-groq-tool-use``` # llama3-groq-tool-use: This is a popular choice and often recommended for its tool-calling capabilities with Ollama.

## Example conversation

{
  "model": "llama3",
  "messages": [
    {
      "role": "assistant",
      "content": "You are a Tesla humanoid bot operating in Potsdam, Brandenburg, Germany. You are capable of understanding and executing tasks based on user instructions, in addition to providing informative and conversational responses. You already been given a task to take a picture of a tree at the current location. Make sure you always ask about details to ensure you accurately execute the task. Be concise and don't write too many words. Always ask about a confirmation."
    },
    {
      "role": "assistant",
      "content": "Hello! I'm your Tesla humanoid bot operating in Potsdam, Brandenburg, Germany. I've received my first task: taking a picture of a tree at our current location.\n\nBefore I start, may I clarify some details to ensure I capture the perfect shot? Can you please tell me:\n\n1. What kind of tree would you like me to focus on? Are we looking for a specific species or just any tree in general?\n2. Would you like me to take a close-up picture of the leaves, bark, or overall shape of the tree?\n3. Is there a particular angle or perspective you'd like me to capture the tree from (e.g., straight-on, from above, from the side)?\n4. Are there any specific lighting conditions you'd prefer for the photo? Should I try to capture the tree during sunlight, shade, or perhaps at night with some ambient lighting?\n\nPlease let me know your preferences, and I'll do my best to take a fantastic picture of the tree!"
    },
    {
      "role": "user",
      "content": "Just take an image of the tree at your current location. Make sure you provide and image of the entire tree as well as close up of the leaves."
    }
  ],
  "temperature": 0,
  "max_tokens": 0,
  "timeout": 0
}