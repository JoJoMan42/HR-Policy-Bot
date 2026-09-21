import os
import uuid
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import bcrypt
import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import (
    load_embedder,
    load_llm,
    load_documents_from_pdf,
    build_chromadb,
    HRAgent,
    PDF_PATH,
)
from database import (
    User,
    conversation_belongs_to_user,
    create_user,
    get_conversation_owner,
    get_conversation_history,
    get_user_by_email,
    get_user_by_id,
    init_db,
)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 168
bearer_scheme = HTTPBearer()

# ----------------------------------------------
# LIFESPAN — initialise once, reuse on every request
# ----------------------------------------------
agent_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SERVER] Initialising PostgreSQL...")
    init_db()
    print("[SERVER] Initialising HR Agent...")
    embedder   = load_embedder()
    llm        = load_llm()
    documents  = load_documents_from_pdf(PDF_PATH)
    collection = build_chromadb(documents, embedder)
    agent_state["agent"] = HRAgent(llm, embedder, collection)
    print("[SERVER] HR Agent ready. Listening for requests.")
    yield
    agent_state.clear()
    print("[SERVER] Shutdown complete.")


# ----------------------------------------------
# APP
# ----------------------------------------------
app = FastAPI(
    title      = "HR Policy Bot API",
    version    = "1.0.0",
    lifespan   = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ----------------------------------------------
# SCHEMAS
# ----------------------------------------------
class ChatRequest(BaseModel):
    message:   str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer:       str
    route:        str
    faithfulness: float
    sources:      list[str]
    thread_id:    str
    user_name:    str | None = None


class AuthRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


def _normalise_email(email: str) -> str:
    normalised = email.strip().lower()
    if "@" not in normalised or normalised.startswith("@") or normalised.endswith("@"):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return normalised


def _create_access_token(user: User) -> str:
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is required. Add it to your .env file.")
    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode({"sub": str(user.id), "exp": expires_at}, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> User:
    if not JWT_SECRET_KEY:
        raise HTTPException(status_code=500, detail="JWT authentication is not configured.")
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User account no longer exists.")
    return user


# ----------------------------------------------
# ROUTES
# ----------------------------------------------
@app.get("/health")
def health():
    ready = "agent" in agent_state
    return {"status": "ok" if ready else "initialising", "agent_ready": ready}


@app.post("/api/auth/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(body: AuthRequest):
    email = _normalise_email(body.email)
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    password_hash = bcrypt.hashpw(body.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        user = create_user(email, password_hash)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return AuthResponse(access_token=_create_access_token(user), user=UserResponse(id=user.id, email=user.email))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(body: AuthRequest):
    email = _normalise_email(body.email)
    user = get_user_by_email(email)
    if user is None or not bcrypt.checkpw(body.password.encode("utf-8"), user.password_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    return AuthResponse(access_token=_create_access_token(user), user=UserResponse(id=user.id, email=user.email))


@app.get("/api/conversations/{thread_id}/history")
def get_history(thread_id: str, current_user: User = Depends(get_current_user)):
    if not conversation_belongs_to_user(thread_id, current_user.id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"thread_id": thread_id, "messages": get_conversation_history(thread_id, current_user.id)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest, current_user: User = Depends(get_current_user)):
    agent = agent_state.get("agent")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent is still initialising. Please retry.")

    thread_id = body.thread_id or str(uuid.uuid4())
    owner_id = get_conversation_owner(thread_id)
    if owner_id is not None and owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        result = agent.ask(body.message, thread_id=thread_id, user_id=current_user.id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer       = result.get("answer", "Sorry, I could not generate a response."),
        route        = result.get("route", "N/A"),
        faithfulness = result.get("faithfulness", 0.0),
        sources      = result.get("sources", []),
        thread_id    = thread_id,
        user_name    = result.get("user_name"),
    )
