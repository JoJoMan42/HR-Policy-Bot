import { useCallback, useEffect, useRef, useState } from 'react'
import AuthScreen from './components/AuthScreen'
import ChatMessage from './components/ChatMessage'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const TOKEN_KEY = 'hrbot_auth_token'
const USER_KEY = 'hrbot_user'

const initialMessages = [{
  id: 'welcome-1', role: 'bot',
  text: 'Hi there! I am KampusBot, your AI campus companion. Ask me anything about attendance criteria, course syllabus, exam schedules, grading rules, or campus facilities.',
  time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), sources: ['Campus Knowledge Base'],
}]

const quickQuestions = [
  { title: 'Attendance Rule', question: 'What is the minimum attendance requirement?' },
  { title: 'CSE Syllabus', question: 'Tell me about the Computer Science & Engineering course structure.' },
  { title: 'SGPA Calculator', question: 'Calculate my SGPA if my grade points are 8.5, 9, 8, 7.5, 9' },
  { title: 'Exam Schedule', question: 'When do the end-semester exams start?' },
  { title: 'Library Policy', question: 'How many books can students borrow from the library?' },
]

function readStoredUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') } catch { return null }
}

function ChatApp({ user, onLogout }) {
  const threadKey = `hrbot_thread_id_${user.id}`
  const [threadId] = useState(() => {
    const existingThreadId = localStorage.getItem(threadKey)
    if (existingThreadId) return existingThreadId
    const newThreadId = crypto.randomUUID()
    localStorage.setItem(threadKey, newThreadId)
    return newThreadId
  })
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState(initialMessages)
  const [isLoading, setIsLoading] = useState(false)
  const [isBackendOnline, setIsBackendOnline] = useState(true)
  const messagesEndRef = useRef(null)

  function authenticatedFetch(path, options = {}) {
    return fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { ...options.headers, Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY)}` },
    })
  }

  const handleUnauthorized = useCallback((response) => {
    if (response.status !== 401) return false
    onLogout()
    return true
  }, [onLogout])

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, isLoading])

  useEffect(() => {
    fetch(`${API_BASE_URL}/health`).then((res) => res.json()).then(() => setIsBackendOnline(true)).catch(() => setIsBackendOnline(false))
  }, [])

  useEffect(() => {
    authenticatedFetch(`/api/conversations/${threadId}/history`)
      .then((res) => {
        if (handleUnauthorized(res)) return null
        return res.ok ? res.json() : null
      })
      .then((data) => {
        if (!data?.messages?.length) return
        setMessages(data.messages.map((message) => ({
          id: `history_${message.id}`, role: message.role === 'user' ? 'student' : 'bot', text: message.content,
          time: new Date(message.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), sources: message.sources || [],
        })))
      }).catch(() => {})
  }, [threadId, handleUnauthorized])

  async function sendMessage(questionText) {
    const cleanQuestion = questionText.trim()
    if (!cleanQuestion || isLoading) return
    const nowTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    setMessages((previous) => [...previous, { id: `student_${crypto.randomUUID()}`, role: 'student', text: cleanQuestion, time: nowTime }])
    setDraft('')
    setIsLoading(true)
    try {
      const response = await authenticatedFetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: cleanQuestion, thread_id: threadId }) })
      if (handleUnauthorized(response)) return
      if (!response.ok) throw new Error(`Server returned ${response.status}`)
      const data = await response.json()
      setIsBackendOnline(true)
      setMessages((previous) => [...previous, {
        id: `bot_${crypto.randomUUID()}`, role: 'bot', text: data.answer || "I couldn't find an answer in the campus documents.",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), sources: data.sources || [],
      }])
    } catch {
      setIsBackendOnline(false)
      setMessages((previous) => [...previous, {
        id: `bot_err_${crypto.randomUUID()}`, role: 'bot', text: `Could not reach the backend server. Please make sure FastAPI is running on ${API_BASE_URL}.`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), sources: [],
      }])
    } finally { setIsLoading(false) }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark" aria-hidden="true">K</div><div className="brand-text"><strong>KampusBot</strong><span>Campus Query Assistant</span></div></div>
        <div className="topbar-right"><div className={`status-badge ${isBackendOnline ? 'status-online' : 'status-offline'}`}><span className="status-dot"></span>{isBackendOnline ? 'Connected' : 'Offline'}</div><button className="logout-button" onClick={onLogout}>Sign out</button></div>
      </header>
      <main className="chat-container">
        <div className="chat-messages-container"><div className="chat-messages-inner"><div className="message-list">
          {messages.map((message) => <ChatMessage key={message.id} {...message} />)}
          {isLoading && <ChatMessage role="bot" text="" time="Thinking..." isTyping={true} />}
          <div ref={messagesEndRef} />
        </div></div></div>
        <div className="composer-wrapper">
          <div className="quick-chips-bar" aria-label="Suggested questions">{quickQuestions.map((item) => <button key={item.title} type="button" className="chip-btn" onClick={() => sendMessage(item.question)} disabled={isLoading}>{item.title}</button>)}</div>
          <form className="composer" onSubmit={(event) => { event.preventDefault(); sendMessage(draft) }}>
            <label className="sr-only" htmlFor="message-input">Ask KampusBot</label>
            <input id="message-input" value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask a campus question..." autoComplete="off" disabled={isLoading} />
            <button type="submit" disabled={!draft.trim() || isLoading}>{isLoading ? '...' : 'Ask'}</button>
          </form>
        </div>
      </main>
    </div>
  )
}

function App() {
  const [user, setUser] = useState(readStoredUser)
  function logout() {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    setUser(null)
  }
  return user ? <ChatApp key={user.id} user={user} onLogout={logout} /> : <AuthScreen onAuthenticated={setUser} />
}

export default App
