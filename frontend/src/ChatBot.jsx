import React, { useState } from "react";

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState();

  const handleSend = async () => {
    if (input.trim() === "") return;

    const userMessage = { text: input, sender: "user" };
    setMessages([...messages, userMessage]);

    const response = await fetch("http://127.0.0.1:5000/chat-bot", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: input }),
    });

    const data = await response.json();
    const botMessage = { text: data.answer, sender: "bot" };
    setMessages((prevMessages) => [...prevMessages, botMessage]);

    setInput("");
  };

  return (
    <div>
      <div>
        {messages.map((message, index) => (
          <div
            key={index}
            style={{ textAlign: message.sender === "user" ? "right" : "left" }}
          >
            {message.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => {
          if (e.key === "Enter") handleSend();
        }}
      />

      <br />

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button
        onClick={async () => {
          if (!file) return;

          const formData = new FormData();
          formData.append("file", file);

          const response = await fetch("http://127.0.0.1:5000/upload-file", {
            method: "POST",
            body: formData,
          });

          const data = await response.json();
          const botMessage = { text: data.answer, sender: "bot" };
          setMessages((prevMessages) => [...prevMessages, botMessage]);

          setFile(null);
        }}
      >
        Send File
      </button>
      <button onClick={handleSend}>Send</button>
    </div>
  );
}

export default ChatBot;
