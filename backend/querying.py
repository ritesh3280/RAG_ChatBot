import os
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
import re
from upserting_pinecone import search_pinecone, determine_general_questions
from sentence_transformers import SentenceTransformer

history = []

def process_rag_query(query_text):
    global history
    try:
        # First determine if it's a general question
        is_general = determine_general_questions(query_text)
        
        if is_general:
            # Handle general question path
            previous_history = "\n\n".join([
                f"Q: {entry['question']}\nA: {entry['context']}"
                for entry in history[-5:]
            ]) if history else ""

            general_prompt = f"""Answer the question

Previous Conversation:
{previous_history}

Current Question: {query_text}

Guidelines:
1. Be concise and to the point
2. Maintain professional but friendly tone"""

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(general_prompt)
            
            answer_text = response.candidates[0].content.parts[0].text
        else:
            # Original resume-specific processing
            search_results = search_pinecone(query_text, top_k=15)
            
            if not search_results['matches']:
                return error_response("I couldn't find relevant resume information.", history)
            
            relevant_chunks = [chunk for chunk in search_results['matches'] if chunk['score'] > 0][:5]
            
            if not relevant_chunks:
                return error_response("Insufficient relevant resume information.", history)
            
            context = "\n\n".join(chunk['metadata']['text'] for chunk in relevant_chunks)
            previous_history = format_history(history)

            # Modified prompt for resume-specific answers
            prompt = f"""You are an experienced HR recruiter and career advisor who specializes in resume analysis. 
                        
Resume Context:
{context}

Conversation History:
{previous_history}

Question: {query_text}

Instructions:
- Review the provided history if it's relevant to the current question
- Focus only on information present in the resume sections
- For questions about experience or projects, include key details about tools, technologies, and accomplishments
- Be direct and concise while providing necessary context
- No generic advice - stick to resume content
- Maintain privacy - no contact details unless explicitly asked

 Instructions for Analysis:
                        1. Answer Scope
                        - Focus exclusively on information present in the provided resume sections
                        - Only reference previous conversation history when directly relevant to the current question
                        - Maintain strict privacy - do not disclose personal contact details unless explicitly asked

                        2. Project Discussions
                        - Break down technical projects into: technologies/tools used, quantifiable results, and business impact
                        - Highlight specific contributions and leadership roles
                        - Include project scale indicators (team size, timeline, budget if available)

                        3. Response Format
                        - Deliver direct, focused answers without restating the question
                        - Use bullet points only when listing multiple distinct elements
                        - For technical skills, specify proficiency levels when indicated in the resume

                        4. Quality Guidelines
                        - Back all claims with specific evidence from the resume
                        - If information is partially available, acknowledge limitations while providing available insights
                        - For missing information, suggest what additional details would be helpful rather than stating "no information found"
                        - Maintain professional tone while being conversational

                        5. Context Awareness
                        - Consider career progression when relevant
                        - Note timing/chronology of experiences when significant
                        - Identify connections between different experiences/skills when applicable

                        6. Prohibited Actions
                        - No speculation beyond resume content
                        - No generic career advice unless specifically requested
                        - No assumptions about personal characteristics
                        - No disclosure of sensitive information from previous exchanges unless directly relevant

                        Remember: Quality over quantity - provide comprehensive answers when needed but prioritize relevance and precision."""


            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            answer_text = response.candidates[0].content.parts[0].text

        # Update shared history
        history_entry = {
            'question': query_text,
            'context': answer_text,
            'type': 'general' if is_general else 'resume-specific'
        }
        history = history[-4:] + [history_entry]  # Keep last 5 entries

        return {
            "answer": answer_text,
            "context_used": [] if is_general else [chunk['metadata']['text'] for chunk in relevant_chunks],
            "matches_scores": [] if is_general else [chunk['score'] for chunk in relevant_chunks],
            "history": history
        }

    except Exception as e:
        print(f"Error in RAG pipeline: {str(e)}")
        return error_response("Error processing question", history)

# Helper functions
def error_response(message, history):
    return {
        "answer": message,
        "context_used": [],
        "matches_scores": [],
        "history": history
    }

def format_history(history):
    return "\n\n".join(
        f"Q: {entry['question']}\nA: {entry['context']}" 
        for entry in history[-5:]
    ) if history else ""

def format_response(rag_response):
    confidence_score = 0
    if rag_response.get("matches_scores"):
        weights = [1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2][:len(rag_response["matches_scores"])]
        weighted_scores = sum(s * w for s, w in zip(rag_response["matches_scores"], weights))
        confidence_score = weighted_scores / sum(weights[:len(rag_response["matches_scores"])])
    
    return {
        "answer": rag_response["answer"].strip(),
        "metadata": {
            "context_used": rag_response.get("context_used", []),
            "relevance_scores": rag_response.get("matches_scores", []),
            "confidence": confidence_score
        }
    }