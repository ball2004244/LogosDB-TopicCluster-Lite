import os
import time
import torch
import warnings
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="transformers.tokenization_utils_base")

'''
Do summarize with QLora fine-tune Flan T5-base
'''

# Define the model and tokenizer directories
MODEL_DIR: str = 'models'
BASE_MODEL_NAME: str = "google/flan-t5-base"
BASE_MODEL_DIR: str = os.path.join(MODEL_DIR, 'flan-t5-base')
TOKENIZER_DIR: str = os.path.join(MODEL_DIR, 'flan-t5-tokenizer')
PEFT_MODEL_NAME: str = 'RMWeerasinghe/flan-t5-base-finetuned-QLoRA-v2'
PEFT_MODEL_DIR: str = os.path.join(MODEL_DIR, 'qlora-flan-t5-model')

# Ensure model and tokenizer are saved locally
if not os.path.exists(BASE_MODEL_DIR):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    base_model.save_pretrained(BASE_MODEL_DIR)

if not os.path.exists(TOKENIZER_DIR):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(TOKENIZER_DIR)

if not os.path.exists(PEFT_MODEL_DIR):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
    model = PeftModel.from_pretrained(base_model, PEFT_MODEL_DIR)
    model.save_pretrained(PEFT_MODEL_DIR)

# Define the DocumentDataset class


class DocumentDataset(Dataset):
    def __init__(self, documents: List[str]) -> None:
        self.documents: List[str] = documents
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        document: str = self.documents[idx]
        inputs: torch.Tensor = self.tokenizer.encode(
            "summarize to 4 sentences: " + document, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        return inputs.squeeze(0)

# Define the inference_batch function


def inference_batch(model: torch.nn.Module, tokenizer: AutoTokenizer, batch: torch.Tensor, device: str) -> List[str]:
    inputs: torch.Tensor = batch.to(device)
    base_model = model.base_model
    output: torch.Tensor = base_model.generate(inputs, max_new_tokens=512,
                                               num_beams=3, do_sample=True, temperature=0.7)
    summaries: List[str] = [tokenizer.decode(
        o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in output]
    return summaries

# Define the mass_abstract_sum function


def mass_qlora_abstract_sum(docs: List[str], BATCH_SIZE: int = 16) -> List[str]:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS: int = os.cpu_count()
    summaries: List[str] = []
    dataset: DocumentDataset = DocumentDataset(docs)
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Load the base model and the fine-tuned model
    base_model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_DIR)
    model: PeftModel = PeftModel.from_pretrained(
        base_model, PEFT_MODEL_DIR).to(device)

    # Load the tokenizer
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    for batch in dataloader:
        summaries.extend(inference_batch(model, tokenizer, batch, device))

    return summaries


# Main block to test the summarization
if __name__ == "__main__":
    raw_document: str = """
    A black hole is a region of spacetime where gravity is so strong that nothing, not even light and other electromagnetic waves, is capable of possessing enough energy to escape it.[2] Einstein's theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.[3][4] The boundary of no escape is called the event horizon. A black hole has a great effect on the fate and circumstances of an object crossing it, but it has no locally detectable features according to general relativity.[5] In many ways, a black hole acts like an ideal black body, as it reflects no light.[6][7] Quantum field theory in curved spacetime predicts that event horizons emit Hawking radiation, with the same spectrum as a black body of a temperature inversely proportional to its mass. This temperature is of the order of billionths of a kelvin for stellar black holes, making it essentially impossible to observe directly. Objects whose gravitational fields are too strong for light to escape were first considered in the 18th century by John Michell and Pierre-Simon Laplace.[8] In 1916, Karl Schwarzschild found the first modern solution of general relativity that would characterise a black hole. David Finkelstein, in 1958, first published the interpretation of "black hole" as a region of space from which nothing can escape. Black holes were long considered a mathematical curiosity; it was not until the 1960s that theoretical work showed they were a generic prediction of general relativity. The discovery of neutron stars by Jocelyn Bell Burnell in 1967 sparked interest in gravitationally collapsed compact objects as a possible astrophysical reality. The first black hole known was Cygnus X-1, identified by several researchers independently in 1971
    """

    # run single document
    doc: List[str] = [raw_document]
    # start = time.perf_counter()
    # summaries = mass_qlora_abstract_sum(doc)
    # print(summaries)
    # print(f'Finished in {time.perf_counter() - start:.2f} seconds')

    # simulate multiple documents
    start: float = time.perf_counter()
    TOTAL_ROW: int = 128
    BATCH_SIZE: int = 16  # Max viable chunk size is 32 for now


    doc = [raw_document] * TOTAL_ROW
    summaries = mass_qlora_abstract_sum(doc, BATCH_SIZE)

    # with open('debug/qlora_abstract_sum.txt', 'w') as f:
    #     for summary in summaries:
    #         f.write(summary + '\n')
    
    print('---')
    print(f'Finished in {time.perf_counter() - start:.2f} seconds')
