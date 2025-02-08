import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine
import asyncio
from sqlalchemy.sql import text

DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/recommender_db"

async def load_csv_to_db():
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    df = pd.read_csv("sports_articles.csv")

    async with engine.begin() as conn:
        for _, row in df.iterrows():
           sql = text(
                "INSERT INTO news_articles (news_id, title, general_category, abstract) VALUES (:news_id, :title, :general_category, :abstract)"
            )
           await conn.execute(sql, {
                "news_id": row["news_id"],
                "title": row["title"],
                "general_category": row["general_category"],
                "abstract": row["abstract"]
            })
    
    await engine.dispose()

asyncio.run(load_csv_to_db())
