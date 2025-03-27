import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import GPT4All
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from typing import TypedDict

# Initialize FastAPI app
app = FastAPI(title="Agentic API", description="Agentic Workflow with FastAPI and LangGraph", version="1.0")

# Define agent state
class AgentState(TypedDict):
    user_input: str
    action: str
    response: str

# Pydantic model for API request
class UserQuery(BaseModel):
    user_input: str

# Initialize local LLM
llm = GPT4All(model="mistral-7b-instruct-v0.1.Q4_K_S.gguf")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
if os.path.exists("faiss_index"):
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded from file.")
else:
    # Load vector database (FAISS example)
    docs = [
        Document(page_content="Basic Plan gives you 10 pages per month for $0.99, with additional pages at $0.10 each."),
        Document(page_content="To upgrade your plan, visit the subscription page in your account settings and select a higher-tier plan."),
        Document(page_content="For billing issues, please check your payment method, update your billing details, or contact support for assistance."),
        Document(page_content="To check your subscription status, visit your account dashboard where you can view current plan details and usage."),
        Document(page_content="The Pro Plan offers unlimited ink refills, priority customer support, and 50% cost savings."),
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)

    # Use a local embedding model
    # Load vector database (FAISS example)
    docs = [
        Document(page_content="Basic Plan gives you 10 pages per month for $0.99, with additional pages at $0.10 each."),
        Document(page_content="To upgrade your plan, visit the subscription page in your account settings and select a higher-tier plan."),
        Document(page_content="For billing issues, please check your payment method, update your billing details, or contact support for assistance."),
        Document(page_content="To check your subscription status, visit your account dashboard where you can view current plan details and usage."),
        Document(page_content="The Pro Plan offers unlimited ink refills, priority customer support, and 50% cost savings."),
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = text_splitter.split_documents(docs)

    # Use a local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_index")
    print("FAISS index saved to file.")


# Intent detection function
def detect_intent(state: AgentState):
    user_query = state["user_input"]
    response = llm.invoke(f"Classify this query: '{user_query}'. Options: [Check Status, Upgrade Plan, Billing Issue, General Inquiry]")
    print(f"Detected action: {response}")
    action = response.split(":", 1)[1].strip().strip()
    return {"action": action, "user_input": user_query}

# Retrieval function
def retrieve_info(state: AgentState):
    query = state["user_input"]
    docs = vector_db.similarity_search(query, k=1)
    answer = docs[0].page_content if docs else "Sorry, I couldn’t find relevant information."
    return {"response": answer}

# Define LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent)
workflow.add_node("retrieve_info", retrieve_info)

def route_action(state: AgentState):
    if state["action"] in ("General Inquiry", "Upgrade Plan", "Billing Issue", "Check Status"):
        return "retrieve_info"
    return None

workflow.add_conditional_edges("detect_intent", route_action)
workflow.set_entry_point("detect_intent")
workflow.set_finish_point("retrieve_info")
app_workflow = workflow.compile()

# FastAPI endpoint
@app.post("/query", summary="Process user query", response_model=dict)
def process_query(user_query: UserQuery):
    output = app_workflow.invoke({"user_input": user_query.user_input})
    return output
