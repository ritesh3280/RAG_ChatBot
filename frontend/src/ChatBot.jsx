import React, { useState } from "react";

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState();
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (input.trim() === "") return;

    // Add user message to chat
    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);
    setIsLoading(true);

    try {
      // Make RAG query instead of regular chat
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Add bot response with context
      const botMessage = {
        text: data.answer,
        sender: "bot",
        context: data.metadata?.context_used,
        relevanceScores: data.metadata?.relevance_scores,
      };

      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      // Handle errors gracefully
      const errorMessage = {
        text: "Sorry, I encountered an error. Please try again.",
        sender: "bot",
        isError: true,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
      setInput("");
    }
  };

  const handleFileUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload-file", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      const botMessage = {
        text: `Successfully processed file: ${file.name}. You can now ask questions about its contents.`,
        sender: "bot",
      };

      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      const errorMessage = {
        text: "Error uploading file. Please try again.",
        sender: "bot",
        isError: true,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setFile(null);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "20px" }}>
      <div
        style={{
          height: "400px",
          overflowY: "auto",
          border: "1px solid #ccc",
          padding: "10px",
          marginBottom: "20px",
        }}
      >
        {messages.map((message, index) => (
          <div
            key={index}
            style={{
              textAlign: message.sender === "user" ? "right" : "left",
              marginBottom: "10px",
              padding: "8px",
              backgroundColor:
                message.sender === "user" ? "#e3f2fd" : "#f5f5f5",
              borderRadius: "8px",
              maxWidth: "70%",
              marginLeft: message.sender === "user" ? "auto" : "0",
            }}
          >
            <div>{message.text}</div>
            {/* Optional: Show context if available */}
            {message.context && (
              <details style={{ fontSize: "0.8em", marginTop: "5px" }}>
                <summary style={{ cursor: "pointer" }}>
                  Show source context
                </summary>
                {message.context.map((ctx, idx) => (
                  <div key={idx} style={{ margin: "5px 0", color: "#666" }}>
                    <div>
                      Relevance: {message.relevanceScores[idx]?.toFixed(2)}
                    </div>
                    <div>{ctx}</div>
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: "10px", marginBottom: "10px" }}>
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ flex: "1" }}
        />
        <button
          onClick={handleFileUpload}
          style={{
            padding: "8px 15px",
            backgroundColor: "#4caf50",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Upload File
        </button>
      </div>

      <div style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === "Enter") handleSend();
          }}
          style={{
            flex: "1",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
          }}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        <button
          onClick={handleSend}
          disabled={isLoading}
          style={{
            padding: "8px 15px",
            backgroundColor: "#2196f3",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: isLoading ? "not-allowed" : "pointer",
            opacity: isLoading ? 0.7 : 1,
          }}
        >
          {isLoading ? "Thinking..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default ChatBot;
