import os
import time
import torch
import warnings
from typing import List
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

'''
Do summarize with QLora fine-tune T5-base
'''

# pip install transformers peft torch

# Define the DocumentDataset class
class DocumentDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        inputs = self.tokenizer.encode(
            "summarize to 4 sentences: " + document, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        return inputs.squeeze(0)

# Define the inference_batch function
def inference_batch(model, tokenizer, batch):
    inputs = batch.to('cuda')
    # Extract the underlying model
    base_model = model.base_model
    output = base_model.generate(inputs, max_new_tokens=512, num_beams=3, do_sample=True, temperature=0.7)
    summaries = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in output]
    return summaries

# Define the mass_abstract_sum function
def mass_qlora_abstract_sum(docs: List[str]) -> List[str]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    summaries = []
    dataset = DocumentDataset(docs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Load the base model and the fine-tuned model
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = PeftModel.from_pretrained(base_model, "RMWeerasinghe/flan-t5-base-finetuned-QLoRA-v2").to(device)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    for batch in dataloader:
        summaries.extend(inference_batch(model, tokenizer, batch))

    return summaries

# Main block to test the summarization
if __name__ == "__main__":
    raw_document = """
    A black hole is a region of spacetime where gravity is so strong that nothing, not even light and other electromagnetic waves, is capable of possessing enough energy to escape it.[2] Einstein's theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.[3][4] The boundary of no escape is called the event horizon. A black hole has a great effect on the fate and circumstances of an object crossing it, but it has no locally detectable features according to general relativity.[5] In many ways, a black hole acts like an ideal black body, as it reflects no light.[6][7] Quantum field theory in curved spacetime predicts that event horizons emit Hawking radiation, with the same spectrum as a black body of a temperature inversely proportional to its mass. This temperature is of the order of billionths of a kelvin for stellar black holes, making it essentially impossible to observe directly. Objects whose gravitational fields are too strong for light to escape were first considered in the 18th century by John Michell and Pierre-Simon Laplace.[8] In 1916, Karl Schwarzschild found the first modern solution of general relativity that would characterise a black hole. David Finkelstein, in 1958, first published the interpretation of "black hole" as a region of space from which nothing can escape. Black holes were long considered a mathematical curiosity; it was not until the 1960s that theoretical work showed they were a generic prediction of general relativity. The discovery of neutron stars by Jocelyn Bell Burnell in 1967 sparked interest in gravitationally collapsed compact objects as a possible astrophysical reality. The first black hole known was Cygnus X-1, identified by several researchers independently in 1971
    """

    doc = [raw_document]
    start = time.perf_counter()
    summaries = mass_qlora_abstract_sum(doc)
    print(summaries)
    print(f'Finished in {time.perf_counter() - start:.2f} seconds')
