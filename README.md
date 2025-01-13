# LogosDB-TopicCluster-Lite
A lighter version of TopicCluster, built fully with Python and SQLite

Simplified pipeline: TopicCluster -> SumAI -> SumDB.

To setup SumAI, check out https://github.com/ball2004244/LogosDB-AI-Models.

## Prerequisites
- Python 3.6 or higher
- SQLite3
- Docker

## Installation
1. Clone the repository
2. Install the required packages
```bash
pip install -r requirements.txt
```

3. Install marqo-vectordb
```bash
bash scripts/setup_sumdb.sh
```

## Usage
1. Have a csv dataset in the root directory
2. Modify process_input.py to reformat dataset to LogosDB input
3. Modify the `pipeline.py` file to suit your needs
2. Run the pipeline
```bash
python3 pipeline.py
```

## Benchmark
1. For benchmarking, need additional requirements
- Ollama with a trained model (LLama3, Mixtral, etc.)
2. Change required parameters in benchmark folder. Most parameters are in `constants.py`
3. Run `benchmark_raw.py` for pure running on LLama3.1 8B or `benchmark_rag.py` for running LLama3.1 8B with RAG
```bash
python3 benchmark/benchmark_raw.py
```
or
```bash
python3 benchmark/benchmark_rag.py
```
4. Calculate result statistically
```bash
python3 benchmark/measure.py
```

## Features
- [x] LogosCluster (Data Storage)
- [x] SumDB (VectorDB)
- [X] SumAI (Summarization - Extractive)
- [ ] SumAI (Summarization - Abstractive)
- [X] SmartQuery (Search for relevant documents)
