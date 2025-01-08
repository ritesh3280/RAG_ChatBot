from pinecone import Pinecone
from dotenv import load_dotenv
import os
import google.generativeai as genai
from embeddings_langchain import embed_query
from upserting_pinecone import search_pinecone


def process_rag_query(query_text):
    """
    Process a RAG query through the complete pipeline:
    1. Embed the query
    2. Search Pinecone for relevant context
    3. Format context and query for LLM
    4. Get LLM response
    """
    try:
        # Step 1: Embed the query
        query_embedding = embed_query(query_text)
        
        # Step 2: Search Pinecone with the embedding
        search_results = search_pinecone(query_embedding, top_k=3)  # Get top 3 most relevant chunks
        
        # Step 3: Format context from search results
        context_chunks = [match['metadata']['text'] for match in search_results['matches']]
        context = "\n\n".join(context_chunks)
        
        # Format the combined context and query for the LLM
        prompt = f"""Please provide a detailed answer to the question based on the provided context. 
        If the context doesn't contain relevant information to fully answer the question, 
        acknowledge that and answer with what can be determined from the given context.

        Context:
        {context}

        Question: {query_text}

        Answer:"""
        
        # Step 4: Get LLM response
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate response
        response = model.generate_content(prompt)
        
        return {
            "answer": response.text,
            "context_used": context_chunks,  # Optional: for debugging
            "matches_scores": [match['score'] for match in search_results['matches']]  # Optional: for debugging
        }
        
    except Exception as e:
        print(f"Error in RAG pipeline: {str(e)}")
        raise

def format_response(rag_response):
    """Format the RAG response for API return"""
    return {
        "answer": rag_response["answer"],
        "metadata": {
            "context_used": rag_response["context_used"],
            "relevance_scores": rag_response["matches_scores"]
        }
    }