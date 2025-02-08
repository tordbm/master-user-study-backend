import random
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database import engine
from app.models import Base, NewsArticles
from app.schemas import Article, Categories

from .database import get_db

app = FastAPI()

origins = [
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


@app.post("/articles", response_model=List[Article])
async def fetch_sports_articles(
    categories: Categories, db: AsyncSession = Depends(get_db)
):
    query = select(NewsArticles).filter(
        NewsArticles.general_category.in_(categories.sports)
    )
    result = await db.execute(query)
    articles = result.scalars().all()

    if not articles:
        raise HTTPException(
            status_code=404, detail="No articles found for the given categories"
        )

    elif len(articles) < 6:
        raise HTTPException(status_code=404, detail="Not enough articles in dataset")

    sampled_articles = []
    for category in categories.sports:
        category_articles = [
            article for article in articles if article.general_category == category
        ]
        sampled_articles.extend(
            random.sample(category_articles, min(3, len(category_articles)))
        )

    return [Article(**article.__dict__) for article in sampled_articles]
