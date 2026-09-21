"""PostgreSQL persistence for chat conversations and messages."""

import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Nullable keeps existing local databases compatible; newly created chats always have an owner.
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    user: Mapped["User | None"] = relationship(back_populates="conversations")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.thread_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    sources: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


def _session_factory():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required. Configure a PostgreSQL connection in .env.")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False), engine


SessionLocal, engine = _session_factory()


def init_db() -> None:
    """Create the small persistence schema on first startup."""
    Base.metadata.create_all(bind=engine)
    # create_all does not add columns to a table created before authentication existed.
    with engine.begin() as connection:
        connection.execute(text(
            "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id "
            "INTEGER REFERENCES users(id) ON DELETE CASCADE"
        ))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations (user_id)"
        ))
    print("[DB] PostgreSQL tables verified.")


def get_user_by_email(email: str) -> User | None:
    with SessionLocal() as db:
        return db.query(User).filter(User.email == email.lower()).first()


def get_user_by_id(user_id: int) -> User | None:
    with SessionLocal() as db:
        return db.get(User, user_id)


def create_user(email: str, password_hash: str) -> User:
    try:
        with SessionLocal.begin() as db:
            user = User(email=email.lower(), password_hash=password_hash)
            db.add(user)
            db.flush()
            db.refresh(user)
            return user
    except IntegrityError as exc:
        raise ValueError("An account with this email already exists.") from exc


def get_conversation_owner(thread_id: str) -> int | None:
    with SessionLocal() as db:
        return db.query(Conversation.user_id).filter(Conversation.thread_id == thread_id).scalar()


def conversation_belongs_to_user(thread_id: str, user_id: int) -> bool:
    return get_conversation_owner(thread_id) == user_id


def get_recent_history(thread_id: str, limit: int = 6) -> list[dict[str, str]]:
    with SessionLocal() as db:
        rows = (
            db.query(Message)
            .filter(Message.thread_id == thread_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .all()
        )
        return [{"role": row.role, "content": row.content} for row in reversed(rows)]


def save_chat_turn(
    thread_id: str,
    user_id: int | None,
    user_message: str,
    assistant_answer: str,
    route: str | None,
    faithfulness: float | None,
    sources: list[str] | None,
) -> None:
    """Save one complete user/assistant exchange as a single transaction."""
    with SessionLocal.begin() as db:
        conversation = db.query(Conversation).filter_by(thread_id=thread_id).first()
        if conversation is None:
            conversation = Conversation(thread_id=thread_id, user_id=user_id)
            db.add(conversation)
            db.flush()

        db.add_all([
            Message(thread_id=thread_id, role="user", content=user_message),
            Message(
                thread_id=thread_id,
                role="assistant",
                content=assistant_answer,
                route=route,
                faithfulness=faithfulness,
                sources=sources or [],
            ),
        ])
        conversation.updated_at = func.now()


def get_conversation_history(thread_id: str, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = (
            db.query(Message)
            .join(Conversation, Message.thread_id == Conversation.thread_id)
            .filter(Message.thread_id == thread_id, Conversation.user_id == user_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "route": row.route,
                "faithfulness": row.faithfulness,
                "sources": row.sources or [],
                "created_at": row.created_at,
            }
            for row in rows
        ]
