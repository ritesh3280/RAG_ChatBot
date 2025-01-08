// react router where /chat-bot is routed to ChatBot.jsx
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import ChatBot from "./ChatBot.jsx";
import App from "./App.jsx";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/chat-bot" element={<ChatBot />} />
      </Routes>
    </Router>
  </StrictMode>,
  document.getElementById("root")
);
