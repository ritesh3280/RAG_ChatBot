from pinecone import Pinecone
from dotenv import load_dotenv
import os
import google.generativeai as genai
from embeddings_langchain import embed_query
from upserting_pinecone import search_pinecone
import re

def rerank_results(matches, query_text):
    scored_chunks = []
    for match in matches:
        relevance_score = match['score']
        text_chunk = match['metadata']['text']
        
        # Apply additional ranking factors
        section_boost = 1.0
        if text_chunk.startswith('## '):
            section_boost = 1.2
        
        # Boost chunks containing query terms
        query_terms = set(query_text.lower().split())
        chunk_terms = set(text_chunk.lower().split())
        term_overlap = len(query_terms.intersection(chunk_terms))
        term_boost = 1.0 + (0.1 * term_overlap)
        
        final_score = relevance_score * section_boost * term_boost
        scored_chunks.append({
            'text': text_chunk,
            'score': final_score,
            'original_score': match['score']
        })
    
    return sorted(scored_chunks, key=lambda x: x['score'], reverse=True)

def process_rag_query(query_text):
    try:
        
        query_embedding = embed_query(query_text)
        search_results = search_pinecone(query_embedding, top_k=15)
        
        if not search_results['matches']:
            return {
                "answer": "I couldn't find relevant information in the resume to answer your question.",
                "context_used": [],
                "matches_scores": []
            }
        
        # Rerank and filter results
        reranked_results = rerank_results(search_results['matches'], query_text)
        relevant_chunks = [
            chunk for chunk in reranked_results
            if chunk['score'] > 0.3
        ][:5]
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find sufficiently relevant information to answer your question accurately.",
                "context_used": [],
                "matches_scores": []
            }
        
        # Format context with section headers and metadata
        context = "\n\n".join([
            f"{chunk['text']}" for chunk in relevant_chunks
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
            "context_used": [chunk['text'] for chunk in relevant_chunks],
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
