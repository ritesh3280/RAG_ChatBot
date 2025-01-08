from dotenv import load_dotenv
import os
import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyC342fTHTCeLiM3oB101zKoTyOO6sKQfNY")

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Start the chat with an initial context
context = """ Orbital Finance - 2nd Place at HackPrinceton | React, Flask, RealTime API, Twilio, LangChain Engineered an AI-powered voice assistant using OpenAI’s Real-Time API and Twilio WebSocket for natural financial conversations Developed a Natural Language to SQL pipeline enabling users to query complex customer databases using
conversational language Built a RAG system using Pinecone, LangChain, and PyTorch to transform raw financial data into actionable
insights"""

query = "What is Orbital Finance?"

# Initialize the chat with a user message
chat = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [{"text": context}],
        },  # Provide context in the user message
        {
            "role": "user",
            "parts": [{"text": query}],
        },  # Add the query as a separate user message
    ]
)

# Send the query to the model
response = chat.send_message(query)

# Print the response text
print(response.text)
