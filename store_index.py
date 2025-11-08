from dotenv import load_dotenv
import os
from src.helper import load_pdf__data, text_split, download_hugging_face_embeddings, filter_to_minimal_docs
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore



load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EURI_API_KEY = os.environ.get("EURI_API_KEY")


extracted_data = load_pdf__data(data='Data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()
 
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
            ),
    )
    
index = pc.Index(index_name)


docSearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)