import random
from typing import List

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database import engine
from app.models import Base, NewsArticles
from app.schemas import Article, Categories, RecommendedArticle

from .database import get_db

app = FastAPI()

origins = [
    "*",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def load_csv_to_db():
    df = pd.read_csv("./data/small_sports_articles.csv")

    async with engine.begin() as conn:
        sql = text(
            "INSERT INTO news_articles (news_id, title, general_category, abstract, tf_idf, s_bert) VALUES (:news_id, :title, :general_category, :abstract, :tfidf_vector, :sbert_vector)"
        )
        data = df.to_dict(orient="records")
        await conn.execute(sql, data)

    df = None


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "create index tf_idf_hnsw_idx on news_articles using hnsw (tf_idf vector_cosine_ops);"
            )
        )
        await conn.execute(
            text(
                "create index s_bert_hnsw_idx on news_articles using hnsw (s_bert vector_cosine_ops);"
            )
        )

    await load_csv_to_db()


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

    shuffled_articles = [Article(**article.__dict__) for article in sampled_articles]
    random.shuffle(shuffled_articles)

    return shuffled_articles


@app.get("/recommend", response_model=List[RecommendedArticle])
async def make_recommendations(db: AsyncSession = Depends(get_db)):
    options = ["tf_idf", "s_bert"]

    rand_vector = np.random.rand(10).tolist()
    vector = "[" + ",".join(map(str, rand_vector)) + "]"

    articles = []
    for i in range(len(options)):
        query = text(f"""
            select news_id,
            title, 
            general_category, 
            abstract 
            from news_articles 
            order by {options[i]} <=> :query_vector 
            limit 5;
            """)
        result = await db.execute(query, {"query_vector": vector})
        similar = result.fetchall()
        articles.extend(
            [
                RecommendedArticle(
                    recommender=options[i],
                    news_id=row.news_id,
                    general_category=row.general_category,
                    title=row.title,
                    abstract=row.abstract,
                )
                for row in similar
            ]
        )
    return articles
