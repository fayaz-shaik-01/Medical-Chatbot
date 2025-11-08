from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from euriai.langchain import create_chat_model

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EURI_API_KEY = os.environ.get("EURI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["EURI_API_KEY"] = EURI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docSearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docSearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)



chatModel = create_chat_model(
    api_key=EURI_API_KEY,
    model="gpt-4.1-nano",
    temperature=0.7
)

question_answer_chain = create_stuff_documents_chain(chatModel,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)



@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST", "GET"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Question: {input}")
    response = rag_chain.invoke({"input": input})
    return str(response['answer'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
    
    