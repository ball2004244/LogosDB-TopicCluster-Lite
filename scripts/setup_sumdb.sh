#!/bin/bash

# This script is used to set up SumDB for the first time.
# Currently we are using Marqo.

docker run -d --name marqo-vectdb -p 8882:8882 marqoai/marqo:latest