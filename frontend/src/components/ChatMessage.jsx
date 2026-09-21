import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function ChatMessage({ role, text, time, sources, source, isTyping }) {
  const isStudent = role === 'student'
  const allSources = sources && sources.length > 0 ? sources : source ? [source] : []

  return (
    <article className={`message ${isStudent ? 'message-student' : 'message-bot'}`}>
      <div className="message-avatar" aria-hidden="true">
        {isStudent ? 'P' : 'K'}
      </div>
      <div className="message-main">
        <div className="message-meta">
          <strong>{isStudent ? 'You' : 'KampusBot'}</strong>
          <span>{time}</span>
        </div>
        <div className="message-bubble">
          {isTyping ? (
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          ) : isStudent ? (
            <span className="student-text">{text}</span>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {text}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {!isTyping && allSources.length > 0 && (
          <div className="message-sources-list">
            <span className="source-heading">Sources:</span>
            {allSources.map((src, idx) => (
              <span key={idx} className="message-source-tag">
                📄 {src}
              </span>
            ))}
          </div>
        )}
      </div>
    </article>
  )
}

export default ChatMessage
