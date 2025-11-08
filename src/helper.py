from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone 
from typing import List
from langchain.schema import Document

## Load the Knowledge base (PDF Files Heer!!!)
def load_pdf__data(data) : 
    loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


## Chuncker which chunks the loaded data
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    docs = text_splitter.split_documents(extracted_data)
    return docs


## Downloading the Opensource Embedding models
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
    

## filter to minimal docs
def filter_to_minimal_docs(docs: list[Document]) -> list[Document]:
    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs