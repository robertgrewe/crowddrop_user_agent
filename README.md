# crowddrop_user_agent
This repo contains code pretending to be a user based on an agentic approach.

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

## Models
Download models by running:

```docker exec -it ollama ollama run llama3```