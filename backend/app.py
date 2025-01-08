from flask import Flask
from flask import request
from flask import jsonify
import google.generativeai as genai
from dotenv import load_dotenv
import os
from flask_mail import Mail
from flask_mail import Message
from flask_cors import CORS, cross_origin
from embeddings_langchain import embed_query
from upserting_pinecone import search_pinecone
from upserting_pinecone import initialize_pinecone, upsert_single_document

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
cors = CORS(app, resources={r"/*": {"origins": "*"}})

load_dotenv()


EMAIL_ADDRESS = "ritesh3280@gmail.com"

app.config["MAIL_SERVER"] = os.getenv("MAIL_SMTP", "smtp.gmail.com")
app.config["MAIL_PORT"] = int(os.getenv("MAIL_PORT", 465))
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME", "ritesh3280@gmail.com")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD", "cmew ehfl ewsx pilr")
app.config["MAIL_USE_TLS"] = False
app.config["MAIL_USE_SSL"] = True
mail = Mail(app)

initialize_pinecone()


@app.route("/submit", methods=["POST"])
@cross_origin()
def gfg():
    try:
        values = request.get_json()

        name = values.get("name")
        message = values.get("message")

        if not name or not message:
            return jsonify({"error": "Missing 'name' or 'message' in the request"}), 400

        mail_message = Message(
            subject=f"New message from {name}",
            sender=EMAIL_ADDRESS,
            recipients=["ritesh3280@gmail.com"],
            body=f"Name: {name}\nMessage: {message}",
        )

        mail.send(mail_message)

        filename = f"{name}.txt"

        with open(filename, "w") as new_file:
            new_file.write("Name : " + name + "\n" + "Message : " + message)

        return jsonify({"message": f"Data saved in {filename}"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred."}), 500


chat_history = []

@app.route("/upload-file", methods=["POST"])
@cross_origin()
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file found"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save the file
        upload_path = os.path.join(
            "/Users/ritesh/Documents/winter_projects/1_backend_comms/project/uploads/",
            file.filename
        )
        file.save(upload_path)
        
        # Process only the new file
        upsert_single_document(file.filename)
        
        return jsonify({
            "message": "File uploaded and processed successfully",
            "filename": file.filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat-bot", methods=["POST"])
@cross_origin()
def chat_bot():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    values = request.get_json()
    prompt = values.get("question")

    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    chat = model.start_chat(history=chat_history)
    response = chat.send_message(prompt, stream=True)

    full_response = ""
    for chunk in response:
        if chunk.text:
            chat_history.append({"role": "assistant", "parts": [{"text": chunk.text}]})
            full_response += chunk.text

    if full_response:
        return jsonify({"answer": full_response})
    else:
        return jsonify({"answer": "No response found"})


@app.route("/query", methods=["GET", "POST"])
@cross_origin()
def query():
    try:
        # Log incoming request
        print("Received request:", request.json)

        # Get the query from the request
        values = request.get_json()
        query_text = values.get("query")
        print("Query Text:", query_text)

        if not query_text:
            return jsonify({"error": "Missing 'query' in the request"}), 400

        # Step 1: Embed the query
        query_embedding = embed_query(query_text)  # Use LangChain to embed the query
        print("Query Embedding:", query_embedding)

        # Step 2: Search Pinecone for relevant context
        search_results = search_pinecone(query_embedding)
        print("Search Results:", search_results)

        # Extract relevant text chunks from Pinecone search results
        context = " ".join(
            [match["metadata"]["text"] for match in search_results["matches"]]
        )
        print("Context:", context)

        if not context:
            return jsonify({"answer": "No relevant context found in the database."})

        # Step 3: Use Gemini LLM to generate an answer
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Start the chat with context as system message
        chat = model.start_chat(
            history=[{"role": "system", "parts": [{"text": context}]}]
        )

        # Send the query with context to the model and receive the response
        response = chat.send_message(query_text, stream=True)

        # Collect response in chunks
        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text

        print("Full Response:", full_response)
        return jsonify({"answer": full_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during the query process."}), 500
    

if __name__ == "__main__":
    app.run(debug=True)