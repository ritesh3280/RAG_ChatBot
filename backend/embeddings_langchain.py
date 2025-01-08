from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import pandas as pd
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings

load_dotenv()

documents_dir = "/Users/ritesh/Documents/winter_projects/1_backend_comms/project/uploads/"


def getText(doc_path):
    base_text = ""

    # Open the PDF document
    pdf_reader = PdfReader(doc_path)
    for page in pdf_reader.pages:
        base_text += page.extract_text()

    return base_text

def clean_text(text):
    # Remove extra newlines and leading/trailing spaces
    cleaned_text = text.replace("\n", " ").strip()  # Removing newlines and extra spaces
    return cleaned_text


def chunk_splitters(text):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

gpt4all_embd = GPT4AllEmbeddings()

def get_embeddings(text):
    embed = gpt4all_embd.embed_documents(text)
    return embed

def get_embeddings_with_cleaning(doc_path):
    text = getText(doc_path)  # Extract text from the document
    clean_text_data = clean_text(text)  # Clean the extracted text
    chunks = chunk_splitters(clean_text_data)  # Split the cleaned text into chunks
    embeddings = get_embeddings(chunks)  # Get embeddings for the chunks
    return embeddings

def embed_query(query_text):
    cleaned_query = clean_text(query_text)  # Clean the query
    query_embedding = gpt4all_embd.embed_query(cleaned_query)
    return query_embedding