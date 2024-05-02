import torch
from datasets import Dataset as hfd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config import DATASET_HF_NAME, LLAMA3_CHECKPOINT

# Adapted from HF https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3


def search_topk(
    data: hfd,
    feature_extractor: SentenceTransformer,
    query: str,
    k: int = 3,
    embedding_col: str = "embedding",
):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = feature_extractor.encode(query)  # embed new query
    scores, retrieved_examples = data.get_nearest_examples(  # retrieve results
        embedding_col,
        embedded_query,  # compare our new embedded query with the dataset embeddings
        k=k,  # get only top k results
    )
    return scores, retrieved_examples


def format_prompt(
    prompt: str, retrieved_documents: hfd, k: int, text_col: str = "chunk"
):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = f"Question:{prompt}\nContext:"
    for idx in range(k):
        PROMPT += f"{retrieved_documents[text_col][idx]}\n"
    return PROMPT


# Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Tokenizer & Model
# You must request access to the checkpoints
TOKENIZER = AutoTokenizer.from_pretrained(LLAMA3_CHECKPOINT)
MODEL = AutoModelForCausalLM.from_pretrained(
    LLAMA3_CHECKPOINT,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
)
TERMINATORS = [TOKENIZER.eos_token_id, TOKENIZER.convert_tokens_to_ids("<|eot_id|>")]

DATA = load_dataset(DATASET_HF_NAME)["train"]

TEXT_GENERATION_PIPELINE = pipeline(
    model=MODEL,
    tokenizer=TOKENIZER,
    task="text-generation",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
TEXT_GENERATION_PIPELINE.tokenizer

PIPELINE_INFERENCE_ARGS = {
    "max_new_tokens": 512,
    "eos_token_id": TERMINATORS,
    "do_sample": True,
    "temperature": 0.2,
    "top_p": 0.9,
}
