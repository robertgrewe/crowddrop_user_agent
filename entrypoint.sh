#!/bin/sh
# entrypoint.sh (Content for /usr/local/bin/ollama-startup.sh)
# No chmod +x needed here or in docker-compose.yml command

/bin/ollama serve &

echo 'Waiting for Ollama server to start...'
sleep 10

echo 'Downloading Ollama models...'
ollama pull llama3
ollama pull llama3-groq-tool-use

echo 'Model download complete.'

wait $! # Wait for the ollama serve process to keep the container alive