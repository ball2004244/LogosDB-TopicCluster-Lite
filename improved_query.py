'''
This file contains the logic of an improved SmartQuery function.

The algorithm is as follows:

Part 1: Original Smart Query
1. Query SumDB with the user's query
2. Extract the top-k results from SumDB
3. Use the extracted results to query LogosCluster
4. Get top-k documents from LogosCluster

Part 2: Extended Smart Query
5. Split each document by paragraph
6. Pass each paragraph as vector to temp SumDB
7. Do similarity search on temp SumDB and get top-k results
8. Return the final results to the user
'''

from typing import List, Dict
from collections import defaultdict
from cluster import LogosCluster
from sumdb import SumDB
import json


def improved_query(cluster: LogosCluster, sumdb: SumDB, query_vector: str, top_k: int = 5) -> List[Dict[str, str]]:
    '''
    Perform an original smart query by querying SumDB first, then use the results to query LogosCluster

    Utilizing temp SumDB to search for more relevant and concise information
    '''
    try:
        # Step 1: Query SumDB
        print(f'Querying SumDB with query: {query_vector}')
        sumdb_results = sumdb.query(query_vector, top_k)

        print(f'Extracted {len(sumdb_results)} results from SumDB')

        # Step 2: Extract the top-k results from SumDB
        topic_map = defaultdict(list)
        row_topic_map = defaultdict(str)
        score_map = defaultdict(int)
        summary_map = defaultdict(str)

        for res in sumdb_results:
            topic, row_id, summary, score = res['topic'], res['row_id'], res['summary'], res['_score']
            topic_map[topic].append(row_id)
            row_topic_map[row_id] = topic
            score_map[row_id] = score
            summary_map[row_id] = summary

        # print(f'Extracted info: {topic_map}')

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
                    'RowID': row_topic_map[_id],
                    'Summary': summary_map[_id],
                    'Content': content,
                    'Score': score_map[_id],
                    'UpdatedAt': updated_at
                })

            count += 1

        # Step 4: Get top-k documents and sort by score
        # sort result by score descending
        cluster_results = sorted(
            cluster_results, key=lambda x: x['Score'], reverse=True)

        print(f'Extracted {len(cluster_results)} results from LogosCluster')

        #* Extended Smart Query
        print(f'START EXTENDED SMART QUERY')
        # Step 5: Split each document by paragraph
        # Set up temp SumDB
        temp_sumdb = SumDB('localhost', 8883)

        # clear temp SumDB
        print(f'Clean up temp SumDB for new data...')
        temp_sumdb.delete_all()

        # split each document by paragraph
        print(f'Splitting each document by paragraph...')
        WORD_PER_PARAGRAPH = 150 #! Change this to the number of words per paragraph
        para_vectors = []
        para_id = 0
        for cluster_res in cluster_results:
            row_id = cluster_res['RowID']
            content = cluster_res['Content']
            topic = row_topic_map[row_id]
            content = content.split(' ') # split by space

            for i in range(0, len(content), WORD_PER_PARAGRAPH):
                para = ' '.join(content[i:i+WORD_PER_PARAGRAPH])
                para_vectors.append({
                    'para_id': para_id,
                    'summary': para,
                    'topic': topic
                })
                para_id += 1

        print(f'Extracted {len(para_vectors)} paragraphs from LogosCluster')
        
        # Step 6: Pass each paragraph vector to temp SumDB
        temp_log = temp_sumdb.insert(para_vectors)

        temp_docs = temp_sumdb.query_all(top_k=len(para_vectors) + 1)
        print(f'Inserted {len(temp_docs)} paragraphs to temp SumDB')

        # Step 7: Do similarity search on temp SumDB and get top-k results
        relevant_paras = temp_sumdb.query(query_vector, top_k)

        # Step 8: Return the final results to the user
        print(f'Extended smart query done. Extracted {len(relevant_paras)} relevant paragraphs')
        return relevant_paras

    except Exception as e:
        print(f'Error at smart_query: {e}')
        return []


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    query = 'What is fiscal policy?'
    sumdb = SumDB()
    cluster = LogosCluster()
    out_dir = 'debug'

    # Perform smart query
    results = improved_query(cluster, sumdb, query)

    with open(f'{out_dir}/improved_query_results.json', 'w') as f:
        json.dump(results, f)

    print(f'Smart query done in {time.perf_counter() - start} seconds.')

    for i, res in enumerate(results):
        summary = res['summary']
        print(f'Para {i+1}, len: {len(summary)}, {summary[:100]}...')