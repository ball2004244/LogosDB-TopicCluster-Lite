#!/bin/bash

# This script is used to set up SumDB for the first time.
# Currently we are using Marqo.

# Run Marqo CPU
# docker run -d --name marqo-vectdb -p 8882:8882 marqoai/marqo:latest

# Run Marqo GPU
# docker run -d --name marqo-gpu-vectdb --gpus all -p 8882:8882 marqoai/marqo:latest
docker run -d --name marqo-vectdb --gpus all -p 8890:8882 \
  -v /home/tam/tam-workspace/LogosDB-TopicCluster-Lite/vectdb_mount/data:/opt/marqo/data \
  -v /home/tam/tam-workspace/LogosDB-TopicCluster-Lite/vectdb_mount/vespa-logs:/opt/vespa/logs \
  -v /home/tam/tam-workspace/LogosDB-TopicCluster-Lite/vectdb_mount/vespa-var:/opt/vespa/var \
  marqoai/marqo:latest


# Run Ollama with:
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
