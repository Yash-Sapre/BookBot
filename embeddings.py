import PyPDF2
from sentence_transformers import SentenceTransformer
import torch


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

pdf_text = ""
chunks = []
embeddings = []

def reset_doc():
    global pdf_text,chunks,embeddings
    pdf_text = ""
    chunks = []
    embeddings = []

def extract_doc(file):
    global pdf_text
    # pdf_file_obj = open("D:/embeddings.pdf","rb")
    r = PyPDF2.PdfReader(file)
    count =len(r.pages)
    for i in range(count):
        pdf_text += r.pages[i].extract_text()


def convert_to_chunks():
    global chunks
    corpus = pdf_text

    corpus = corpus.split(".")
    total_sent_num=len(corpus)
    chunks = []
    chunk_size = 6
    for i in range(0,total_sent_num - chunk_size):
        chunks.append(".\n".join(corpus[i:i+chunk_size]))

def embed_doc():
    global embeddings
    embeddings = model.encode(chunks)


def semantic_search(query):
    query_embedding = model.encode(query)
    similarity_scores = model.similarity(query_embedding,embeddings)[0]
    score,ind = torch.topk(similarity_scores,k = 1)
    if(score < 0):
        return None
    else:
        return chunks[ind]






