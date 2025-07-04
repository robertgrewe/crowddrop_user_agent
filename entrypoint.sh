#!/bin/bash

# Start Ollama server in the background
/bin/ollama serve &

# Record the PID of the Ollama server process
OLLAMA_PID=$!

echo "Waiting for Ollama server to start..."
# Wait for Ollama to be ready. You can check for port availability or a specific log message.
# A simple sleep is often enough for Docker Compose. Adjust if needed.
sleep 10 # Give Ollama a few seconds to fully initialize

echo "Downloading Ollama models..."

# Download llama3
ollama pull llama3

# Download llama3-groq-tool-use
ollama pull llama3-groq-tool-use

echo "Model download complete."

# Wait for the Ollama server process to finish (it should run indefinitely)
wait $OLLAMA_PID