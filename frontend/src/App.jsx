import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const SAVED_CHATS_KEY = 'lawgpt_saved_chats_v1';

function App() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState('GENERAL');
  const [responseLanguage, setResponseLanguage] = useState('English');
  const [isSending, setIsSending] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploadSummaries, setUploadSummaries] = useState([]);
  const [status, setStatus] = useState('Ready');
  const [savedChats, setSavedChats] = useState([]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(SAVED_CHATS_KEY);
      if (stored) {
        setSavedChats(JSON.parse(stored));
      }
    } catch {
      setSavedChats([]);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(SAVED_CHATS_KEY, JSON.stringify(savedChats));
  }, [savedChats]);

  const conversationPairs = useMemo(() => {
    const pairs = [];
    let pending = null;
    for (const msg of messages) {
      if (msg.role === 'user') pending = msg.content;
      if (msg.role === 'assistant' && pending) {
        pairs.push({
          id: msg.pairId,
          question: pending,
          answer: msg.content,
        });
        pending = null;
      }
    }
    return pairs;
  }, [messages]);

  const saveChat = (chat) => {
    setSavedChats((prev) => {
      const exists = prev.some((item) => item.id === chat.id);
      if (exists) return prev.filter((item) => item.id !== chat.id);
      return [{ ...chat }, ...prev].slice(0, 20);
    });
  };

  const isSaved = (id) => savedChats.some((item) => item.id === id);

  const resetAllChats = () => {
    setMessages([]);
    setQuestion('');
    setStatus('Ready');
  };

  const sendQuestion = async () => {
    const text = question.trim();
    if (!text || isSending) return;

    const pairId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const userMessage = { role: 'user', content: text, pairId };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setIsSending(true);
    setStatus('Generating answer...');

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          response_language: responseLanguage,
          mode,
        }),
      });

      if (!res.ok) throw new Error(`Chat API failed: ${res.status}`);

      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || 'No answer returned.',
          pairId,
        },
      ]);
      setStatus('Answer loaded');
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Unable to load answer. Please ensure API server is running on port 8000.',
          pairId,
        },
      ]);
      setStatus('Error while fetching answer');
    } finally {
      setIsSending(false);
    }
  };

  const uploadDocs = async () => {
    if (!files.length || isUploading) return;
    setIsUploading(true);
    setStatus('Uploading and indexing documents...');

    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);

      const data = await res.json();
      setUploadSummaries(data.summaries || []);
      setStatus(`Processed ${data.processed_files || 0} files`);
    } catch (err) {
      setStatus('Upload failed. Check API server logs.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="history-pane">
        <div className="history-pill">SAVED CHATS</div>
        <h3>Bookmarked answers</h3>
        {savedChats.length === 0 && <p className="empty-note">Save any answer to keep it here.</p>}
        {savedChats.map((item) => (
          <div className="history-card saved" key={item.id}>
            <div className="history-card-top">
              <span className="history-badge">Saved</span>
              <button className="ghost-button" onClick={() => saveChat(item)} title="Unsave this answer">
                Unsave
              </button>
            </div>
            <div className="history-q">{item.question}</div>
            <div className="history-a">{item.answer}</div>
          </div>
        ))}
      </aside>

      <main className="main-pane">
        <header className="hero">
          <div className="kicker">⚖️ LAW CHAMBER</div>
          <h1>LawGPT Legal Assistant</h1>
          <p>Justice-oriented chat for legal guidance, document intake, and quick responses.</p>
        </header>

        <section className="panel">
          <h2>Document Intake</h2>
          <input
            type="file"
            multiple
            accept=".pdf,.jpg,.jpeg,.png"
            onChange={(e) => setFiles(Array.from(e.target.files || []))}
          />
          <button onClick={uploadDocs} disabled={isUploading || files.length === 0}>
            {isUploading ? 'Processing...' : 'Process Files'}
          </button>
          {uploadSummaries.length > 0 && (
            <div className="summaries">
              {uploadSummaries.map((s, i) => (
                <details key={`s-${i}`}>
                  <summary>{s.file_name} ({s.word_count} words)</summary>
                  <div className="summary-lines">
                    <p className="summary-paragraph">{String(s.summary || '')}</p>
                  </div>
                </details>
              ))}
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Legal Q&A</h2>
          <div className="controls">
            <label>
              Response Language
              <select value={responseLanguage} onChange={(e) => setResponseLanguage(e.target.value)}>
                <option>English</option>
                <option>Hindi</option>
                <option>Telugu</option>
              </select>
            </label>
            <label>
              Ask Questions From
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option>GENERAL</option>
                <option>DOCUMENT</option>
              </select>
            </label>
          </div>

          <div className="chat-box">
            {messages.map((m, idx) => (
              <div key={`m-${idx}`} className={`bubble ${m.role}`}>
                {m.role === 'assistant' && (
                  <div className="bubble-head">
                    <span className="response-icon">⚖️</span>
                    <span className="response-label">Legal Response</span>
                    <span className="response-accent">📜</span>
                    <button
                      className={`save-button ${isSaved(m.pairId) ? 'saved' : ''}`}
                      onClick={() =>
                        saveChat({
                          id: m.pairId,
                          question: messages[idx - 1]?.content || 'Question',
                          answer: m.content,
                        })
                      }
                      disabled={!m.pairId}
                      title={isSaved(m.pairId) ? 'Unsave' : 'Save answer'}
                    >
                      {isSaved(m.pairId) ? '★ Saved' : '☆ Save'}
                    </button>
                  </div>
                )}
                <div className={`bubble-text ${m.role === 'user' ? 'text-user' : 'text-assistant'}`}>{m.content}</div>
              </div>
            ))}
          </div>

          <div className="composer">
            <textarea
              placeholder="Ask a legal question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendQuestion();
                }
              }}
            />
            <button className="send-button" onClick={sendQuestion} disabled={isSending || !question.trim()}>
              <span className="send-icon">⚖️</span>
              <span>{isSending ? 'Sending...' : 'Send to Court'}</span>
            </button>
          </div>

          <div className="panel-footer-actions">
            <button className="reset-button" onClick={resetAllChats} disabled={messages.length === 0}>
              Reset All Chats
            </button>
          </div>
        </section>

        <footer className="status-bar">Status: {status}</footer>
      </main>
    </div>
  );
}

export default App;
