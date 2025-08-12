import React, { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage = { sender: 'user', text: question };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setIsLoading(true);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const aiMessage = { sender: 'ai', text: data.answer, source: data.source_url };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);

    } catch (error) {
      console.error("Failed to fetch from API:", error);
      const errorMessage = { sender: 'ai', text: `Sorry, I encountered an error: ${error.message}` };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    } finally {
      setQuestion('');
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>CivicData AI</h1>
        <p>Ask me anything about Ukrainian open data.</p>
      </header>
      <div className="chat-window">
        <div className="message-list">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <p>{msg.text}</p>
              {msg.source && (
                <a href={msg.source} target="_blank" rel="noopener noreferrer" className="source-link">
                  Source
                </a>
              )}
            </div>
          ))}
          {isLoading && <div className="message ai"><p><i>Thinking...</i></p></div>}
        </div>
        <form onSubmit={handleSubmit} className="chat-form">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., How many new companies were registered in Kyiv last year?"
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;