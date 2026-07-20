import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Render se environment variable uthao
DATABASE_URL = os.environ.get("DATABASE_URL")

# Agar URL postgres:// se start ho rahi hai, toh use postgresql:// me badal do
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Agar local testing ke liye URL na mile, toh fallback string (Aapki Neon URL)
if not DATABASE_URL:
    DATABASE_URL = "abc"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()  # <--- Ye line hona bahut zaroori hai!

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
