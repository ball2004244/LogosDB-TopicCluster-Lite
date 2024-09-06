from typing import List, Dict
from extract_sum_mp import mass_extract_summaries
from cluster import LogosCluster
import marqo

'''
This file contains the SumDB class, which is responsible for storing summarized vectors and querying similar content.
'''


class SumDB:
    def __init__(self):
        self.host = 'localhost'
        self.port = 8882

        # set up Marqo connector
        self.db = marqo.Client(url=f'http://{self.host}:{self.port}')

        # Only create index if it does not exist
        index_name = 'sumdb'
        try:
            self.db.create_index(index_name, model="hf/e5-base-v2")
        except:
            pass
        self.index = self.db.index(index_name)

    def insert(self, vectors: List[Dict[str, str]], CHUNK_SIZE: int = 128) -> bool:
        '''
        Insert summarized vectors into SumDB in chunks

        NOTE that maximum chunk size is 128

        Vector data structure:
        {
            'row_id': 'unique_id',
            'summary': 'content_text',
            'topic': 'topic_name',
        }
        '''
        try:
            for i in range(0, len(vectors), CHUNK_SIZE):
                batch = vectors[i:i + CHUNK_SIZE]
                self.index.add_documents(batch, tensor_fields=['summary'])

            return True

        except Exception as e:
            print(f'Error at SumDB insert: {e}')
            return False

    def query(self, query_vector: str, top_k: int = 5) -> List[Dict[str, str]]:
        '''
        Query similar content from SumDB, only output top results with highest similarity score
        '''
        try:
            return self.index.search(
                q=query_vector,
                limit=top_k
            )['hits']

        except Exception as e:
            print(f'Error at SumDB query: {e}')
            return []

    def query_all(self, top_k: int = 5) -> List[Dict[str, str]]:
        '''
        Query all content from SumDB, only output top results with highest similarity score
        '''
        try:
            return self.index.search(
                q='*',
                limit=top_k
            )['hits']

        except Exception as e:
            print(f'Error at SumDB query_all: {e}')
            return []

    def delete_all(self) -> bool:
        '''
        Delete all vectors from SumDB by actual ID, not row_id
        '''
        try:
            all_docs = self.index.search(
                q='', limit=128)  # Adjust limit as needed
            while len(all_docs['hits']) > 0:
                id_set = []
                for hit in all_docs['hits']:
                    id_set.append(hit['_id'])
                print(f'Deleting {len(id_set)} documents')
                self.index.delete_documents(id_set)
                
                all_docs = self.index.search(q='', limit=128)

            return True

        except Exception as e:
            print(f'Error at SumDB delete_all: {e}')
            return False

    def summarize_node(self, node: str, cluster: LogosCluster, CHUNK_SIZE: int = 128) -> bool:
        '''
        Summarize content from a single node in the cluster
        '''
        try:
            # Divide data into chunks to avoid memory overload
            count = 0
            for chunk in cluster.query_chunk(node, CHUNK_SIZE):
                insert_data = []
                print(f'[INFO] Summarizing chunk {count}...')
                # Each chunk is a list of rows (ID: int, Content: str, UpdatedAt: datetime)
                # For each chunk, summarize the content
                summaries = mass_extract_summaries([row[1] for row in chunk])

                # Then modified the chunk with summarized content
                for summary, row in zip(summaries, chunk):
                    insert_data.append(
                        {'row_id': row[0], 'summary': summary, 'topic': node})

                # Finally insert summarized data into SumDB
                self.insert(insert_data, CHUNK_SIZE)

                print(f'[INFO] Finished summarizing chunk {count}')
                count += 1

            return True

        except Exception as e:
            print(f'Error at SumDB summarize_node: {e}')
            return False

    def summarize_cluster(self, cluster: LogosCluster, CHUNK_SIZE: int = 128) -> bool:
        '''
        Summarize all content from the cluster to SumDB
        '''
        try:
            # Query data from each node and summarize
            for node in cluster.nodes:
                print(f'[INFO] Processing node {node}')
                insert_status = self.summarize_node(node, cluster, CHUNK_SIZE)

                if not insert_status:
                    return False

                print(f'[INFO] Node {node} summarized successfully')

            return True

        except Exception as e:
            print(f'Error at SumDB summarize_cluster: {e}')
            return False


if __name__ == '__main__':
    import time
    from rich import print
    start = time.perf_counter()
    sumdb = SumDB()
    print('Inserting data...')
    # sumdb.insert(
    #     [
    #         {'row_id': '1', 'summary': 'This is a test', 'topic': 'test'},
    #         {'row_id': '2', 'summary': 'This is another test', 'topic': 'test'},
    #         {'row_id': '3', 'summary': 'This is 2nd testing scenario', 'topic': 'test'},
    #     ]
    # )
    query = 'agnostida extinct order of arthropods agnostida is an order of arthropod which have classically been seen as a group of highly modified trilobites'
    print('Querying data...')
    print(sumdb.query(query))

    print('Querying all data...')
    print(sumdb.query_all())

    print('Final cleanup...')
    sumdb.delete_all()

    print(f'Total time: {time.perf_counter() - start:.2f} seconds')
