'''
This file contains the logic of smart query function on AuxiDB

The algorithm is as follows:
1. Query SumDB with the user's query
2. Extract the top-k results from SumDB
3. Return directly the results to the user
'''

from typing import List, Dict
from sumdb import SumDB


def auxi_query(sumdb: SumDB, query_vector: str, top_k: int = 5) -> List[Dict[str, str]]:
    '''
    Perform a smart query by querying SumDB first, then use the results to query LogosCluster
    '''
    try:
        # Step 1: Query SumDB
        # print(f'Querying SumDB with query: {query_vector}')
        sumdb_results = sumdb.query(query_vector, top_k)

        # print(f'Extracted {len(sumdb_results)} results from SumDB')

        # Step 2: Extract the top-k results from SumDB
        output = []
        for res in sumdb_results:
            topic, row_id, summary, score = res['topic'], res['row_id'], res['summary'], res['_score']
            output.append({
                'ID': row_id,
                'Summary': summary,
                'Score': score
            })

        # sort result by score descending
        output = sorted(
            output, key=lambda x: x['Score'], reverse=True)

        return output

    except Exception as e:
        print(f'Error at auxi_query: {e}')
        return []


if __name__ == '__main__':
    import time
    import datetime

    print(
        f'START testing Auxi Query at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    start = time.perf_counter()
    query = 'What is blackhole?'

    port = 8884  # for AuxiDB
    # port = 8885 # for AuxiLogosExtract
    # port = 8890 # for AuxiLogosQloraAbstract
    auxi_db = SumDB(port=port)
    out_dir = 'debug'

    # Perform multi query
    n = 5000
    for i in range(n):
        if (i + 1) % 100 == 0:
            print(f'Processing query {i+1}/{n}')
        results = auxi_query(auxi_db, query)

        # with open(f'{out_dir}/auxi_query_results.json', 'w') as f:
        #     json.dump(results, f)

        if (i + 1) % 1000 == 0:
            print(f'Finished query {i+1}/{n}')

    print(
        f'Auxi query done in {time.perf_counter() - start} seconds.(~{(time.perf_counter() - start) / 60:.2f} minutes)')

    print(
        f'FINISHED Auxi Query at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
