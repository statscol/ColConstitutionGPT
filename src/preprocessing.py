import textract
from datasets import Dataset as hfd
from config import FEATURE_EXTRACTOR_CHECKPOINT
from sentence_transformers import SentenceTransformer

FEATURE_EXTRACTOR=SentenceTransformer(FEATURE_EXTRACTOR_CHECKPOINT)


def encode_sentence(instance: hfd,text_col:str):

    return {'embedding': FEATURE_EXTRACTOR.encode(instance[text_col],normalize_embeddings=True)}


def parse_pdf(pdf_path: str):
    """Gets text from a pdf file using textract"""
    txt = textract.process(
        pdf_path,
        method='pdfminer',
        encoding='latin-1').decode()
    return txt

def chunk_text(text: str,split_sentence="ARTÍCULO"):
    """creates chunks of texts using a split_sentence"""
    chunks=[{'chunk':split_sentence + " " + c.replace('\n', ' ').strip()}
            for c in text.split(split_sentence)]
    print(len(chunks))
    return chunks

def create_df(text_chunks:list[dict[str]]):
    "creates a HuggingFace dataset based on a list of dicts [str,str]"
    return hfd.from_list(text_chunks)



if __name__=="__main__":

    #push to hub pipe
    text=parse_pdf("./data/Reforma-pensional.pdf")
    chunks=chunk_text(text)
    df=create_df(chunks)
    df=df.map(encode_sentence,batched=True,fn_kwargs={'text_col':'chunk'})
    df.push_to_hub("jhonparra18/reforma-pensional-col")