from common import DATA, MODEL, TERMINATORS, TOKENIZER, format_prompt, search_topk
from config import MAX_TOKENS_INPUT, SYS_PROMPT_HF
from preprocessing import FEATURE_EXTRACTOR


def generate(formatted_prompt):
    formatted_prompt = formatted_prompt[:MAX_TOKENS_INPUT]  # to avoid GPU OOM
    messages = [
        {"role": "system", "content": SYS_PROMPT_HF},
        {"role": "user", "content": formatted_prompt},
    ]

    input_ids = TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(MODEL.device)
    outputs = MODEL.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=TERMINATORS,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
    )
    response = outputs[0]
    return TOKENIZER.decode(response[input_ids.shape[-1] :], skip_special_tokens=True)


def rag_chatbot(prompt: str, k: int = 2):
    _, retrieved_documents = search_topk(
        DATA, FEATURE_EXTRACTOR, prompt, k, embedding_col="embedding"
    )
    formatted_prompt = format_prompt(prompt, retrieved_documents, k, text_col="chunk")
    return f"[USER]: {prompt}\n\n[ASSISTANT]: {generate(formatted_prompt)}"


if __name__ == "__main__":
    # example RAG Pipeline using HuggingFace
    DATA = DATA.add_faiss_index("embedding")
    prompt = """indicame si en la reforma pensional qué va a pasar con los fondos en
      el pilar contributivo de prima media, podré pedir el dinero de vuelta?"""
    print(rag_chatbot(prompt, k=3))
