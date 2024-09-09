'''
This file contains the logic of smart query function.

The algorithm is as follows:
1. Query SumDB with the user's query
2. Extract the top-k results from SumDB
3. Use the extracted results to query LogosCluster
4. Return the final results to the user
'''

from typing import List, Dict
from collections import defaultdict
from cluster import LogosCluster
from sumdb import SumDB
import json

def smart_query(cluster: LogosCluster, sumdb: SumDB, query_vector: str, top_k: int = 5) -> List[Dict[str, str]]:
    '''
    Perform a smart query by querying SumDB first, then use the results to query LogosCluster
    '''
    try:
        # Step 1: Query SumDB
        print(f'Querying SumDB with query: {query_vector}')
        sumdb_results = sumdb.query(query_vector, top_k)
        
        print(f'Extracted {len(sumdb_results)} results from SumDB')

        # Step 2: Extract the top-k results from SumDB
        topic_map = defaultdict(list)
        score_map = defaultdict(int)
        for res in sumdb_results:
            topic, row_id, score = res['topic'], res['row_id'], res['_score']
            topic_map[topic].append(row_id)
            score_map[row_id] = score

        print(f'Extracted info: {topic_map}')

        # Step 3: Use the extracted results to query LogosCluster
        print('Querying LogosCluster by each topic node')
        cluster_results = []
        count = 0
        for topic, row_ids in topic_map.items():
            print(f'Node {count}: {topic}, row_ids: {row_ids}')
            results = cluster.query_by_ids(row_ids, topic)

            # match the score with result
            for res in results:
                _id, content, updated_at = res
                cluster_results.append({
                    'ID': _id,
                    'Content': content,
                    'UpdatedAt': updated_at,
                    'Score': score_map[_id]
                })            

            count += 1
            
        # sort result by score descending
        cluster_results = sorted(cluster_results, key=lambda x: x['Score'], reverse=True)
        return cluster_results
        
    except Exception as e:
        print(f'Error at smart_query: {e}')
        return []
    
if __name__ == '__main__':
    query = 'What is quantum gravity and its role in modern astronomy?'    
    sumdb = SumDB()
    cluster = LogosCluster()
    out_dir = 'outputs'

    # Perform smart query
    results = smart_query(cluster, sumdb, query)
    
    with open(f'{out_dir}/smart_query_results.json', 'w') as f:
        json.dump(results, f)