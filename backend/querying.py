from pinecone import Pinecone
from dotenv import load_dotenv
import os
import google.generativeai as genai
from embeddings_langchain import embed_query
from upserting_pinecone import search_pinecone
import re

def process_rag_query(query_text):
    try:
        search_results = search_pinecone(query_text, top_k=15)
        
        if not search_results['matches']:
            return {
                "answer": "I couldn't find relevant information in the resume to answer your question.",
                "context_used": [],
                "matches_scores": []
            }
        
        relevant_chunks = []
        for chunk in search_results['matches']:
            print(chunk['score'])
            if chunk['score'] > 0.1:
                relevant_chunks.append(chunk)

        relevant_chunks = relevant_chunks[:5]
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find sufficiently relevant information to answer your question accurately.",
                "context_used": [],
                "matches_scores": []
            }
        
        # Format context with section headers and metadata
        context = "\n\n".join([
            f"{chunk['metadata']['text']}" for chunk in relevant_chunks
        ])
        
        prompt = f"""You are a professional resume analyzer. Answer the following question based on the resume sections provided:

Context from Resume:
{context}

Question: {query_text}

Requirements:
- Answer specifically based on the provided resume sections
- If the information isn't in the context, say so
- Be concise and direct
- Focus only on relevant details from the resume"""
        
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        
        return {
            "answer": response.text,
            "context_used": [chunk['metadata']['text'] for chunk in relevant_chunks],
            "matches_scores": [chunk['score'] for chunk in relevant_chunks]
        }
        
    except Exception as e:
        print(f"Error in RAG pipeline: {str(e)}")
        raise

def format_response(rag_response):
    confidence_score = 0
    if rag_response["matches_scores"]:
        weights = [1.2, 1.0, 0.8, 0.6, 0.4][:len(rag_response["matches_scores"])]
        weighted_scores = sum(s * w for s, w in zip(rag_response["matches_scores"], weights))
        confidence_score = weighted_scores / sum(weights[:len(rag_response["matches_scores"])])
    
    return {
        "answer": rag_response["answer"],
        "metadata": {
            "context_used": rag_response["context_used"],
            "relevance_scores": rag_response["matches_scores"],
            "confidence": confidence_score
        }
    }


