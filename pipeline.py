import os
import numpy as np
import tiktoken
import pickle
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI

# Load .env variables
load_dotenv()

# Constants
MODEL_ENCODING_NAME = "cl100k_base"
MAX_CONTEXT_TOKENS_FOR_LLM = 12000
MAX_CHUNK_SIZE_TOKENS = 500
OVERLAP_TOKENS = 50
KNOWLEDGE_BASE_FILE = "knowledge_base.pkl"

# Initialize tokenizer
encoding = tiktoken.get_encoding(MODEL_ENCODING_NAME)

# Azure Document Intelligence
di_client = DocumentIntelligenceClient(
    endpoint=os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("DOCUMENTINTELLIGENCE_API_KEY"))
)

# Azure OpenAI
openai_client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
chat_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Helpers
def get_embedding(text):
    response = openai_client.embeddings.create(input=text, model=embedding_model)
    return np.array(response.data[0].embedding)

def chunk_text(text, source_doc_name):
    tokens = encoding.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i: i + MAX_CHUNK_SIZE_TOKENS]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({"text_content": chunk_text, "source_doc_name": source_doc_name})
        i += (MAX_CHUNK_SIZE_TOKENS - OVERLAP_TOKENS)
    return chunks

def extract_general_document_data(analyze_result):
    return {"content": analyze_result.content or ""}

def process_uploaded_file(file_bytes):
    poller = di_client.begin_analyze_document("prebuilt-layout", body=file_bytes)
    result = poller.result()
    text = extract_general_document_data(result)["content"]
    chunks = chunk_text(text, "uploaded_file")
    return [{
        "text": c["text_content"],
        "embedding": get_embedding(c["text_content"]),
        "source": c["source_doc_name"]
    } for c in chunks]

def run_rag_query(kb, query):
    query_embedding = get_embedding(query)
    similarities = [(np.dot(query_embedding, item["embedding"]), item) for item in kb]
    similarities.sort(key=lambda x: x[0], reverse=True)

    context = []
    total_tokens = 0
    for _, item in similarities:
        token_count = len(encoding.encode(item["text"]))
        if total_tokens + token_count <= MAX_CONTEXT_TOKENS_FOR_LLM:
            context.append(item["text"])
            total_tokens += token_count
        else:
            break

    context_string = "\n\n".join(context)
    prompt = f"Context:\n{context_string}\n\nQuestion: {query}"

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for analyzing RFP/RFQ documents."},
        {"role": "user", "content": prompt}
    ]

    response = openai_client.chat.completions.create(
        messages=messages,
        model=chat_model,
        temperature=0.2,
        max_tokens=1000
    )
    return response.choices[0].message.content

def load_base_kb():
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return []
    with open(KNOWLEDGE_BASE_FILE, "rb") as f:
        kb = pickle.load(f)
    for item in kb:
        if not isinstance(item["embedding"], np.ndarray):
            item["embedding"] = np.array(item["embedding"])
    return kb