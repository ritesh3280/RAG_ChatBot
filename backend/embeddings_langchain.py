from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain_community.embeddings import GPT4AllEmbeddings
import re

load_dotenv()

documents_dir = "/Users/ritesh/Documents/winter_projects/1_backend_comms/project/uploads/"

def getText(doc_path):

    try:
        if doc_path.lower().endswith('.pdf'):
            pdf_reader = PdfReader(doc_path)
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    
                    text_parts.append(text.strip())
            
            raw_text = '\n\n'.join(text_parts)
        else:
            with open(doc_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
        
       
        markdown_text = detect_and_format_headers(raw_text)
        return markdown_text
        
    except Exception as e:
        print(f"Error extracting text from {doc_path}: {str(e)}")
        raise


def detect_and_format_headers(text):
    lines = text.split('\n')
    formatted_lines = []
    current_section = []
    
    resume_sections = [
        "Education", "Experience", "Work History", "Skills", 
        "Projects", "Certifications", "Achievements", 
        "Publications", "References"
    ]

    # Picks out lines that are section headers - WORKING(Tested)
    def is_resume_section_header(line):
        line = line.strip()
        contains_keyword = any(section.lower() in line.lower() for section in resume_sections)

        is_title_like = line.isupper() or line.istitle()

        is_not_bullet_point = not line.startswith(("-", "*", "•", "1.", "a."))

        is_standalone = len(line.split()) <= 7

        return contains_keyword and is_title_like and is_not_bullet_point and is_standalone
    
    # Formats the text into MARKDOWN with symbols - WORKING(Tested)
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
            formatted_lines.append(f"## {line}")
        else:
            current_section.append(line)
        
    if current_section:
        formatted_lines.extend(current_section)

    return '\n'.join(formatted_lines)

def chunk_splitters(text):
    headers_to_split_on = [
        ("##", "Section"),
    ]
    
    markdown_splitters = MarkdownHeaderTextSplitter(headers_to_split_on)
    md_header_splits = markdown_splitters.split_text(text)
    
    chunks = []
    for chunk in md_header_splits:
        try:
            page_content = getattr(chunk, "page_content")
            section = chunk.metadata.get('Section', None)
            if page_content:
                chunks.append({"page_content": page_content, "section": section})
            else:
                print("No page content")
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")

    return chunks

gpt4all_embd = GPT4AllEmbeddings()

def get_embeddings(texts):
    embeddings = []
    try:
        for chunk in texts:
            page_content = chunk.get('page_content', '')

            embedding = gpt4all_embd.embed_documents([page_content])
            embeddings.append(embedding)

        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def embed_query(query_text):
    return gpt4all_embd.embed_query((query_text))

def process_document(doc_path):
    try:

        markdown_text = getText(doc_path)
        chunks = chunk_splitters(markdown_text)
        embeddings = get_embeddings(chunks)
        
        return embeddings, chunks
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise


# get_embeddings(chunk_splitters(getText("/Users/ritesh/Documents/winter_projects/1_backend_comms/project/uploads/Ritesh_Thipparthi_Resume_2027.pdf")))
