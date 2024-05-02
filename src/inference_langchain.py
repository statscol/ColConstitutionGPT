from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever

# from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from common import DATA, TEXT_GENERATION_PIPELINE, PIPELINE_INFERENCE_ARGS
from config import FEATURE_EXTRACTOR_CHECKPOINT, LANGC_PROMPT_TEMPLATE

TEXT_GENERATION_PIPELINE.tokenizer.pad_token_id = (
    TEXT_GENERATION_PIPELINE.tokenizer.eos_token_id
)
llm = HuggingFacePipeline(
    pipeline=TEXT_GENERATION_PIPELINE, pipeline_kwargs=PIPELINE_INFERENCE_ARGS
)

db = FAISS.from_texts(
    DATA["chunk"], HuggingFaceEmbeddings(model_name=FEATURE_EXTRACTOR_CHECKPOINT)
)

# Use the top-k most relevant docs
retriever = db.as_retriever(search_kwargs={"k": 3})
parser = StrOutputParser()

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=LANGC_PROMPT_TEMPLATE,
)


rag_chain = prompt | llm | parser


def run_qa(chain: RunnableSequence, retriever: VectorStoreRetriever, question: str):
    docs = retriever.invoke(question)
    response = rag_chain.invoke({"context": docs, "question": question})
    response = f"Question: {question}\n\n Answer: {response.split('Respuesta:')[-1]}"
    return response


if __name__ == "__main__":
    question = """Indicame en la reforma pensional qué va a pasar con los fondos
      en el pilar contributivo de prima media, podré pedir el dinero de vuelta?"""
    print(run_qa(rag_chain, retriever, question))
