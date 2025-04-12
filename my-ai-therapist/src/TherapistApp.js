import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import "./TherapistApp.css"; // Import styles

export default function TherapistApp() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [emotion, setEmotion] = useState("Neutral");
  const [isLoading, setIsLoading] = useState(false);
  const webcamRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setIsLoading(true);

    const userMessage = { sender: "You", text: input, emotion };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input, emotion }),
      });

      const data = await response.json();
      console.log("Response:", data);
      const botMessage = { sender: "Therapist", text: data.response, emotion };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setInput("");
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:5000/emotion");
        const data = await res.json();
        setEmotion(data.emotion);
      } catch (err) {
        console.error("Emotion fetch failed");
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-container">
      <div className="video-section">
        <h1 className="title">AI Therapist</h1>
        <Webcam audio={false} ref={webcamRef} className="webcam" />
        <p className="emotion-text">
          Detected Emotion: <strong>{emotion}</strong>
        </p>
      </div>

      <div className="chat-section">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`card ${msg.sender === "You" ? "user" : "bot"}`}>
              <div className="card-content">
                <p className="sender">
                  {msg.sender} ({msg.emotion}):
                </p>
                <p>{msg.text}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="input-section">
          <input
            className="input-box"
            placeholder="Talk to your therapist..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          />
          <button className="send-button" onClick={sendMessage} disabled={isLoading}>
            {isLoading ? "Thinking..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
