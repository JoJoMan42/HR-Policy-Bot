import { useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setIsSubmitting(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/${mode}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Authentication failed.')
      localStorage.setItem('hrbot_auth_token', data.access_token)
      localStorage.setItem('hrbot_user', JSON.stringify(data.user))
      onAuthenticated(data.user)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setIsSubmitting(false)
    }
  }

  const isSignup = mode === 'signup'
  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-mark" aria-hidden="true">K</div>
        <p className="auth-eyebrow">KampusBot</p>
        <h1 id="auth-title">{isSignup ? 'Create your account' : 'Welcome back'}</h1>
        <p className="auth-subtitle">{isSignup ? 'Save your own private chat history.' : 'Sign in to continue to your assistant.'}</p>
        <form className="auth-form" onSubmit={handleSubmit}>
          <label htmlFor="auth-email">Email address</label>
          <input id="auth-email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required />
          <label htmlFor="auth-password">Password</label>
          <input id="auth-password" type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={isSignup ? 'new-password' : 'current-password'} minLength="8" required />
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button type="submit" className="auth-submit" disabled={isSubmitting}>{isSubmitting ? 'Please wait…' : isSignup ? 'Create account' : 'Sign in'}</button>
        </form>
        <button type="button" className="auth-switch" onClick={() => { setMode(isSignup ? 'login' : 'signup'); setError('') }}>
          {isSignup ? 'Already have an account? Sign in' : 'New here? Create an account'}
        </button>
      </section>
    </main>
  )
}

export default AuthScreen
