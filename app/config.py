import os

from dotenv import load_dotenv
from sqlalchemy import URL

load_dotenv()

DATABASE_URL = (
    URL.create(
        "postgresql+asyncpg",
        username=os.getenv("DATABASE_USER"),
        password=os.getenv("DATABASE_PASSWORD"),
        host=os.getenv("DATABASE_HOST"),
        database=os.getenv("DATABASE_NAME"),
    )
    if os.getenv("ENVIRONMENT") == "production"
    else os.getenv("DATABASE_URL")
)

PROLIFIC_CODE = os.getenv("PROLIFIC_CODE")

REDIRECT_URL = os.getenv("REDIRECT_URL")
