from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime

# === DATABASE CONFIGURATION ===
DATABASE_URL = "sqlite+aiosqlite:///./predictions.db"  # Switch to PostgreSQL if needed

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

# === ORM MODEL ===
class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    predicted_class = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# === DEPENDENCY FOR FASTAPI ===
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

import asyncio

# === DATABASE INITIALIZATION ===
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created.")

if __name__ == "__main__":
    asyncio.run(init_db())
