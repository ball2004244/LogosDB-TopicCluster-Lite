# LogosDB-TopicCluster-Lite

A lighter version of TopicCluster, built fully with Python and SQLite

Simplified pipeline: TopicCluster -> SumAI -> SumDB.

To setup SumAI, check out <https://github.com/ball2004244/LogosDB-AI-Models>. Or running:

## Prerequisites

- Python 3.6 or higher
- SQLite3
- Docker

## Installation

1. Clone the repository
2. Create conda environment

    ```bash
    conda env create -f environment.yml
    ```

3. Install marqo-vectordb

    ```bash
    bash scripts/setup_sumdb.sh
    ```

## Usage

1. Have a csv dataset in the root directory
2. Modify process_input.py to reformat dataset to LogosDB input
3. Modify the `scripts/pipeline.py` file to suit your needs
4. Run the pipeline

    ```bash
    python3 -m scripts.pipeline
    ```

## Benchmark

1. For benchmarking, need additional requirements: Ollama with a trained model (LLama3, Mixtral, etc.)

2. Change required parameters in benchmark folder. Most parameters are in `constants.py`
3. Run `scripts/multi_benchmark.py` for benchmarking LogosDB on MMLU datasets

    ```bash
    python3 -m scripts.multi_benchmark
    ```

## Features

- [X] LogosCluster (Data Storage)
- [X] SumDB (VectorDB)
- [X] SumAI (Summarization - Extractive)
- [X] SumAI (Summarization - Abstractive)
- [X] SmartQuery (Search for relevant documents)
