import json
import requests
from io import BytesIO
import PyPDF2
from bs4 import BeautifulSoup
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_context_from_json(file_path):
    """Load PDF and web URLs from a JSON file."""
    with open(file_path, 'r') as json_file:
        context = json.load(json_file)
    return context["pdf_urls"], context["web_urls"]

def download_and_extract_text(pdf_url):
    """Download a PDF from the URL and extract its text."""
    response = requests.get(pdf_url)
    pdf_content = BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_content)
    
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text() or ""
    
    return text

def scrape_text_from_web(url):
    """Scrape text from a web page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    paragraphs = soup.find_all('p')
    text = " ".join(paragraph.get_text() for paragraph in paragraphs)
    
    return text

# Load URLs from context.json
pdf_urls, web_urls = load_context_from_json('context.json')

# Extract text from both PDFs
combined_text = ""
for url in pdf_urls:
    combined_text += download_and_extract_text(url)

# Extract text from both web pages
for url in web_urls:
    combined_text += scrape_text_from_web(url)

# Load the pre-trained models
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Split text into passages for retrieval
passages = combined_text.split('\n\n')

# Encode passages into embeddings
passage_embeddings = embedder.encode(passages, convert_to_tensor=False)

# Create a FAISS index for efficient retrieval
dimension = passage_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(passage_embeddings))

def retrieve_relevant_passages(question, top_k=3):
    """Retrieve the top_k most relevant passages for a given question."""
    question_embedding = embedder.encode([question], convert_to_tensor=False)
    distances, indices = index.search(np.array(question_embedding), top_k)
    return [passages[idx] for idx in indices[0]]

def answer_question(question):
    """Retrieve relevant context and generate a detailed answer to the given question."""
    relevant_passages = retrieve_relevant_passages(question)
    context = " ".join(relevant_passages)

    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="tf", truncation=True)

    output = model.generate(**inputs)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    detailed_answer = f"**Question:** {question}\n**Answer:** {answer}\n\n"
    return detailed_answer

# Example usage with detailed answers
questions = [
    "What is the main purpose of the HP Instant Ink program and how does it fit into HP's overall strategy?",
    "Can you explain the different types of HP Instant Ink plans available and what differentiates them?",
    "How does the subscription model of HP Instant Ink benefit consumers compared to traditional ink purchasing methods?",
    "What technology does HP use to monitor ink levels and predict when new cartridges are needed?",
    "What specific benefits do users of HP Instant Ink report in terms of convenience and cost savings?",
    "How does HP Instant Ink handle different printing needs, such as high-volume printing or occasional use?",
    "What options do customers have for customizing their HP Instant Ink plans based on individual usage patterns?",
    "What are the terms and conditions regarding changing or canceling an HP Instant Ink subscription?",
    "Are there any specific requirements for HP printers to be compatible with the HP Instant Ink program?",
    "How does the recycling program for HP Instant Ink cartridges work, and what environmental benefits does it provide?",
    "What measures does HP take to ensure customer satisfaction with the Instant Ink program?",
    "How does HP Instant Ink compare to similar subscription services offered by competitors?",
    "What are the limitations or drawbacks of the HP Instant Ink program that customers should be aware of?",
    "How do promotional offers or discounts impact the pricing of HP Instant Ink plans?",
    "What feedback has HP received from users about their experiences with the Instant Ink program?"
]


# Collecting and printing detailed answers for each question
for question in questions:
    answer = answer_question(question)
    print(answer)
