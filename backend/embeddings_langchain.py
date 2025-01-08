from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
import re

load_dotenv()

documents_dir = "/Users/ritesh/Documents/winter_projects/1_backend_comms/project/uploads/"

def getText(doc_path):
    """Extract text from resumes while preserving section headers and structure"""
    try:
        if doc_path.lower().endswith('.pdf'):
            pdf_reader = PdfReader(doc_path)
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    # Preserve structure with double newlines between pages
                    text_parts.append(text.strip())
            
            raw_text = '\n\n'.join(text_parts)
        else:
            with open(doc_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
        
        # Convert to markdown format with section headers
        markdown_text = detect_and_format_headers(raw_text)
        return markdown_text
        
    except Exception as e:
        print(f"Error extracting text from {doc_path}: {str(e)}")
        raise

def detect_and_format_headers(text):
    """Convert resume text to markdown format with proper section headers"""
    lines = text.split('\n')
    formatted_lines = []
    current_section = []
    
    # Common resume section headers
    resume_sections = [
        "Education", "Experience", "Work History", "Skills", 
        "Projects", "Certifications", "Achievements", 
        "Publications", "References"
    ]
    
    def is_resume_section_header(line):
        """Check if a line is a resume section header"""
        line = line.strip()
        return any(
            section.lower() in line.lower() 
            for section in resume_sections
        )
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_section:
                formatted_lines.extend(current_section)
                current_section = []
            formatted_lines.append("")
            continue

        if is_resume_section_header(line):
            if current_section:
                formatted_lines.extend(current_section)
                current_section = []
            formatted_lines.append(f"## {line}")  # Markdown header for sections
        else:
            current_section.append(line)
        
    if current_section:
        formatted_lines.extend(current_section)

    return '\n'.join(formatted_lines)


def clean_text(text):
    """Clean text while preserving semantic structure and meaningful content"""
    # Remove excessive whitespace and non-printable characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    
    # Remove irrelevant metadata (e.g., page numbers, headers/footers)
    text = re.sub(r'Page \d+ of \d+', '', text)  # Remove page numbers
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)  # Remove emails
    
    # Preserve sentence structure
    text = re.sub(r'\.\s+', '. ', text)  # Ensure proper spacing after periods
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Preserve paragraph breaks
    
    return text.strip()

def chunk_splitters(text):
    # Define headers hierarchy for resume sections
    headers_to_split_on = [
        ("##", "Section"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    # First split by headers
    header_splits = header_splitter.split_text(text)
    
    # Then split by size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for split in header_splits:
        chunk = text_splitter.split_text(split.page_content)
        chunks.extend(chunk)
    
    return chunks



def get_embeddings(texts):
    try:
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Get raw vectors directly
            batch_embeddings = gpt4all_embd.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings  # Return just the raw vectors
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise



gpt4all_embd = GPT4AllEmbeddings()

def get_embeddings_with_cleaning(doc_path):
    try:
        markdown_text = getText(doc_path)
        clean_text_data = clean_text(markdown_text)
        text_chunks = chunk_splitters(clean_text_data)
        embeddings = get_embeddings(text_chunks)
        return embeddings, text_chunks
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise

def embed_query(query_text):
    """Embed query with minimal preprocessing"""
    return gpt4all_embd.embed_query(clean_text(query_text))

def process_document(doc_path):
    try:
        # Get text and clean it
        markdown_text = getText(doc_path)
        clean_text_data = clean_text(markdown_text)
        
        # Generate chunks
        chunks = chunk_splitters(clean_text_data)
        
        # Generate embeddings
        embeddings = get_embeddings(chunks)
        
        return embeddings, chunks
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise
