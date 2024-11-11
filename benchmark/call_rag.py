from typing import List
import sys
import os

# Add the absolute path of parent dir to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from the parent directory
from cluster import LogosCluster
from sumdb import SumDB
from smart_query import smart_query
from improved_query import improved_query
from auxi_db_query import auxi_query

'''
Here are caller functions to call different type of RAG models.

#! Query Algorithms:
1. Basic query is SmartQuery, which queries SumDB first, then LogosCluster.
2. ImprovedQuery is better SmartQuery, which queries SumDB first, then LogosCluster, shorten results by paragraphs with ParaDB.
3. Auxiliary Query is traditional query method, which queries SumDB only. (In this case SumDB is full-content VectorDB

#! Note: To run Improved Query, set use_improved_query to True, and you MUST run ParaDB container alongside with SumDB.
'''

def call_rag(query: str, k: int=5, use_improved_query: bool=False) -> List[str]:
    '''
    Helper function to call LogosDB as RAG model.
    '''
    cluster = LogosCluster()
    sumdb = SumDB()

    #! Switch between smart_query and improved_query for different scenarios
    if not use_improved_query:
        results = smart_query(cluster, sumdb, query, k)
    else:
        results = improved_query(cluster, sumdb, query, k)

    # concatenate the results
    output = []
    for res in results:
        output.append(res['Summary'])

    return output

def call_auxi_logos(query: str, k: int=5, use_improved_query: bool=True) -> List[str]:
    '''
    Helper function to call LogosDB as RAG model.
    '''
    cluster = LogosCluster('auxi_logos')
    port = 8885 # 8885 for extract AuxiDB
    # port = 8886 # 8886 for abstract AuxiDB
    # port = 8890 # for Qlora abstract AuxiDB

    sumdb = SumDB('localhost', port)

    #! Switch between smart_query and improved_query for different scenarios
    if not use_improved_query:
        results = smart_query(cluster, sumdb, query, k)
    else:
        results = improved_query(cluster, sumdb, query, k)

    # concatenate the results
    output = []
    for res in results:
        output.append(res['Summary'])

    # remove duplicate
    output = list(set(output))
    return output


def call_auxi_db(query: str, k: int=5) -> List[str]:
    '''
    Helper function to call SumDB as auxiliary training data.
    '''
    sumdb = SumDB('localhost', 8884)

    results = auxi_query(sumdb, query, k)

    # concatenate the results
    output = []
    for res in results:
        # remove result with score < 0.7
        if res['Score'] < 0.7:
            continue
        output.append(res['Summary'])

    return output

def call_multi_rag(query: str, k: int=5) -> List[str]:
    '''
    Helper function that allow setting ups multiple rag

    1. Use original queries to look up relevant docs from AuxiDB
    2. Each AuxiVector serves as intermediate queries to look up SumDB & LogosCluster
    '''
    auxi_results = call_auxi_db(query, k)

    # remove duplicate
    auxi_results = list(set(auxi_results))

    #! NOTE: improved query goes with call_rag so no need to explicitly code here
    logos_results = call_rag(query, k, use_improved_query=True)

    # remove duplicate
    logos_results = list(set(logos_results))

    return logos_results

if __name__ == '__main__':
    import json
    query = "What is a black hole?"
    # res = call_rag(query)
    res = call_auxi_logos(query)
    # res = call_auxi_db(query)
    # res = call_multi_rag(query)

    debug_dir = 'debug'
    
    with open(f'{debug_dir}/call_rag_py.json', 'w') as f:
        json.dump(res, f)
    print(f'Finished writing result to {debug_dir}/call_rag_py.json')
