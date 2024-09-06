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

    def insert(self, vectors: List[Dict[str, str]]) -> bool:
        '''
        Insert summarized vectors into SumDB

        Vector data structure:
        {
            '_id': 'unique_id',
            'summary': 'content_text',
            'topic': 'topic_name',
        }
        '''
        try:
            self.index.add_documents(vectors, tensor_fields=['summary'])
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
        Delete all vectors from SumDB
        '''
        try:
            all_docs = self.index.search(
                q='', limit=400)  # Adjust limit as needed
            while len(all_docs['hits']) > 0:
                id_set = []
                for hit in all_docs['hits']:
                    id_set.append(hit['_id'])
                self.index.delete_documents(id_set)
                all_docs = self.index.search(q='', limit=400)

            return True

        except Exception as e:
            print(f'Error at SumDB delete_all: {e}')
            return False

    def summarize(self, paragraphs: List[str]) -> List[str]:
        '''
        Summarize list of paragraphs
        '''
        summaries = mass_extract_summaries(paragraphs)

        return summaries

    def summarize_node(self, node: str, cluster: LogosCluster) -> List[Dict[str, str]]:
        '''
        Summarize content from a single node in the cluster
        '''
        insert_data = []
        try:
            # Divide data into chunks to avoid memory overload
            for chunk in cluster.query_chunk(node):
                # Each chunk is a list of rows (ID: int, Content: str, UpdatedAt: datetime)
                # For each chunk, summarize the content
                summaries = self.summarize([row[1] for row in chunk])

                # Then modified the chunk with summarized content
                for summary, row in zip(summaries, chunk):
                    insert_data.append(
                        {'_id': row[0], 'summary': summary, 'topic': node})

        except Exception as e:
            print(f'Error at SumDB summarize_node: {e}')

        finally:
            return insert_data

    def summarize_cluster(self, cluster: LogosCluster) -> bool:
        '''
        Summarize all content from the cluster to SumDB
        '''
        try:
            # Query data from each node and summarize
            for node in cluster.nodes:
                insert_data = self.summarize_node(node, cluster)
                # Finally insert the summarized data by chunk
                self.insert(insert_data)

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
    #         {'_id': '1', 'summary': 'This is a test', 'topic': 'test'},
    #         {'_id': '2', 'summary': 'This is another test', 'topic': 'test'},
    #         {'_id': '3', 'summary': 'This is 2nd testing scenario', 'topic': 'test'},
    #     ]
    # )
    query = 'This is a test'
    print('Querying data...')
    print(sumdb.query(query))

    print('Querying all data...')
    print(sumdb.query_all())

    # print('Final cleanup...')
    # sumdb.delete_all()

    print(f'Total time: {time.perf_counter() - start:.2f} seconds')
