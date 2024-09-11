import sys
import os

# Add the absolute path to slm_tune to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
# import from the parent directory
from cluster import LogosCluster
from sumdb import SumDB
from smart_query import smart_query

def call_rag(query: str, k: int=5) -> List[str]:
    '''
    Helper function to call LogosDB as RAG model.
    '''
    cluster = LogosCluster()
    sumdb = SumDB()
    
    results = smart_query(cluster, sumdb, query, k)
    
    # concatenate the results
    output = []
    for res in results:
        output.append(res['Summary'])

    return output

if __name__ == '__main__':
    query = "What is the capital of France?"
    res = call_rag(query)
    print(res)
