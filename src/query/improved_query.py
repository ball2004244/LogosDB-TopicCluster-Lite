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
6. Pass each paragraph as vector to paraDB
7. Do similarity search on paraDB and get top-k results
8. Return the final results to the user
'''

from typing import List, Dict
from collections import defaultdict
from src.core import LogosCluster, SumDB
import json


def improved_query(cluster: LogosCluster, sumdb: SumDB, query_vector: str, top_k: int = 5, verbose: bool = False, para_db_host: str = 'localhost', para_db_port: int = 8883) -> List[Dict[str, str]]:
    '''
    Perform an original smart query by querying SumDB first, then use the results to query LogosCluster

    Utilizing paraDB to search for more relevant and concise information
    '''
    try:
        # Step 1: Query SumDB
        if verbose:
            print(f'Querying SumDB with query: {query_vector}')
        sumdb_results = sumdb.query(query_vector, top_k)

        if verbose:
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

        # Step 3: Use the extracted results to query LogosCluster
        if verbose:
            print('Querying LogosCluster by each topic node')
        cluster_results = []
        count = 0
        for topic, row_ids in topic_map.items():
            if verbose:
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

        if verbose:
            print(
                f'Extracted {len(cluster_results)} results from LogosCluster')

        # * Extended Smart Query
        if verbose:
            print(f'START EXTENDED SMART QUERY')
        # Step 5: Split each document by paragraph
        # Set up paraDB on port 8883 (Split docs into paragraphs)
        paraDB = SumDB(para_db_host, para_db_port)

        # clear paraDB to prepare for new data
        if verbose:
            print(f'Clean up paraDB for new data...')
        paraDB.delete_all()

        # split each document by paragraph
        if verbose:
            print(f'Splitting each document by paragraph...')
        WORD_PER_PARAGRAPH = 150  # ! Change this to the number of words per paragraph
        para_vectors = []
        para_id = 0
        for cluster_res in cluster_results:
            row_id = cluster_res['RowID']
            content = cluster_res['Content']
            topic = row_topic_map[row_id]
            content = content.split(' ')  # split by space

            for i in range(0, len(content), WORD_PER_PARAGRAPH):
                para = ' '.join(content[i:i+WORD_PER_PARAGRAPH])
                para_vectors.append({
                    'para_id': para_id,
                    'summary': para,
                    'topic': topic
                })
                para_id += 1

        if verbose:
            print(
                f'Extracted {len(para_vectors)} paragraphs from LogosCluster')

        # Step 6: Pass each paragraph vector to paraDB
        paraDB.insert(para_vectors)

        docs = paraDB.query_all(top_k=len(para_vectors) + 1)
        if verbose:
            print(f'Inserted {len(docs)} paragraphs to paraDB')

        # Step 7: Do similarity search on paraDB and get top-k paragraphs
        relevant_paras = paraDB.query(query_vector, top_k)

        if verbose:
            print(
                f'Extended smart query done. Extracted {len(relevant_paras)} relevant paragraphs')

        # Step 8: Reformat and Return the final results to the user

        formatted_results = []
        for res in relevant_paras:
            formatted_results.append({
                'ID': res['para_id'],
                'Summary': res['summary'],
                'Topic': res['topic'],
                'Score': res['_score']
            })
        return formatted_results

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

    print(f'Check out return log at {out_dir}/improved_query_results.json')
