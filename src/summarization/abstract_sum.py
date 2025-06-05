import time
import os
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

# pip install transformers torch
'''
Do summarize with t5-base model
'''

class DocumentDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents
        self.tokenizer = T5Tokenizer.from_pretrained('T5-base')

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        inputs = self.tokenizer.encode(
            "summarize: " + document, return_tensors='pt', max_length=512, truncation=True)
        return inputs.squeeze(0)


def inference_batch(model, tokenizer, batch):
    inputs = batch.to('cuda')
    output = model.generate(inputs, min_length=50, max_length=100)
    summaries = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    return summaries


def mass_abstract_sum(docs: List[str]) -> List[str]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    summaries = []
    dataset = DocumentDataset(docs)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model_name = 'T5-base'
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, return_dict=True).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    for batch in dataloader:
        summaries.extend(inference_batch(model, tokenizer, batch))

    return summaries


if __name__ == "__main__":
    raw_document = """
    Nature is a tapestry of life, woven with vibrant hues and textures. From towering mountains to tranquil oceans, each element plays a vital role in this intricate ecosystem. The rustling leaves whisper secrets, the gurgling streams sing melodies, and the sun's warm embrace nourishes all. Nature offers solace to weary souls, inspiring awe and wonder. Its delicate balance reminds us of our interconnectedness and the importance of preserving this precious gift for generations to come.
    """

    doc = [raw_document]
    start = time.perf_counter()
    summaries = mass_abstract_sum(doc)
    print(summaries)
    print(f'Finished in {time.perf_counter() - start:.2f} seconds')
