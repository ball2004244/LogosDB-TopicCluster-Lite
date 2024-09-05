# LogosDB-TopicCluster-Lite
A lighter version of TopicCluster, built fully with Python and SQLite

## Prerequisites
- Python 3.6 or higher
- SQLite3
- Marqo VectorDB

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
1. Modify the `pipeline.py` file to suit your needs
2. Run the pipeline
```bash
python3 pipeline.py
```
