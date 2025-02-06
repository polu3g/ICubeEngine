import os
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from typing import TypedDict

# Define agent state
class AgentState(TypedDict):
    user_input: str
    action: str
    response: str

# Initialize local LLM (Make sure you have the GPT4All model downloaded)
llm = GPT4All(model="ggml-mistral-7b-instruct-q4_0.gguf", n_ctx=512)

# Load vector database (FAISS example)
docs = [
    Document(page_content="The Pro Plan offers unlimited ink refills, priority customer support, and 50% cost savings."),
    Document(page_content="Basic Plan gives you 10 pages per month for $0.99, with additional pages at $0.10 each."),
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Use a local embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(documents, embeddings)

# Intent detection function
def detect_intent(state: AgentState):
    user_query = state["user_input"]
    response = llm.invoke(f"Classify this query: '{user_query}'. Options: [Check Status, Upgrade Plan, Billing Issue, General Inquiry]")
    
    action = response.strip()
    return {"action": action, "user_input": user_query}

# Retrieval function
def retrieve_info(state: AgentState):
    query = state["user_input"]
    docs = vector_db.similarity_search(query, k=1)  # Fetch most relevant document
    answer = docs[0].page_content if docs else "Sorry, I couldn’t find relevant information."
    return {"response": answer}

# Define LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("retrieve_info", retrieve_info)

# Define edges
workflow.add_edge("detect_intent", "retrieve_info", condition=lambda state: state["action"] == "General Inquiry")

# Set entry and finish points
workflow.set_entry_point("detect_intent")
workflow.set_finish_point("retrieve_info")

# Compile graph
app = workflow.compile()

# Example user query
user_input = "What does the Pro Plan include?"
output = app.invoke({"user_input": user_input})

print(output)
