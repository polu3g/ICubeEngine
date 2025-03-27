import pandas as pd
import json


# Sample fleet log data
data = [
    {"printer_id": "HP123", "status": "Low Ink", "location": "Office A", "last_service": "2025-01-15"},
    {"printer_id": "HP456", "status": "Paper Jam", "location": "Office B", "last_service": "2025-02-01"},
]

# Save as JSON
with open("printer_fleet.json", "w") as f:
    json.dump(data, f)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import json

# Load data
with open("printer_fleet.json") as f:
    fleet_logs = json.load(f)

# Convert to text format
documents = [Document(page_content=json.dumps(log)) for log in fleet_logs]

# Create text splitter (if logs are large)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# Generate embeddings & store in FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPEN_API_KEy , openai_organization=OPEN_AI_ORG)
vector_db = FAISS.from_documents(split_docs, embeddings)
vector_db.save_local("fleet_index")
print("Vector database saved to file.")

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

# Load vector DB
vector_db = FAISS.load_local("fleet_index", embeddings)

# LLM setup
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPEN_API_KEy , openai_organization=OPEN_AI_ORG)

# Retrieval chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())

# Function to get insights
def get_printer_insights(query):
    return qa_chain.run(query)

 # Example query
print(get_printer_insights("Which printers have low ink?"))



import streamlit as st

st.title("Printer Fleet Insights")
query = st.text_input("Ask about your fleet:")
if st.button("Get Insights"):
    response = get_printer_insights(query)
    st.write(response)
