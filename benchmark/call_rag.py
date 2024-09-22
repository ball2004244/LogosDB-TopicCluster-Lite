import sys
import os

# Add the absolute path to slm_tune to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
# import from the parent directory
from cluster import LogosCluster
from sumdb import SumDB
from smart_query import smart_query
from improved_query import improved_query
from auxi_query import auxi_query

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

def call_auxi_train(query: str, k: int=5) -> List[str]:
    '''
    Helper function to call SumDB as auxiliary training data.
    '''
    sumdb = SumDB('localhost', 8884)
    
    results = auxi_query(sumdb, query, k)

    # concatenate the results
    output = []
    for res in results:
        output.append(res['Summary'])

    return output

def call_multi_rag(query: str, k: int=5) -> List[str]:
    '''
    Helper function that allow setting ups multiple rag

    1. Use original queries to look up relevant docs from AuxiDB
    2. Each AuxiVector serves as intermediate queries to look up SumDB & LogosCluster
    '''
    auxi_results = call_auxi_train(query, k)

    # remove duplicate
    auxi_results = list(set(auxi_results))

    logos_results = call_rag(query, k)

    # remove duplicate
    logos_results = list(set(logos_results))

    # pass relevant info through ParaDB
    # TODO: Implement ParaDB
    pass

    return logos_results

if __name__ == '__main__':
    query = "What is the capital of France?"
    # res = call_rag(query)
    # print(res)

    res = call_auxi_train(query)
    print(res)
