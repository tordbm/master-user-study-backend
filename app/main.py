import json
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
from app.enums import Recommenders
from app.middleware import RetryMiddleware
from app.models import Base, NewsArticles
from app.schemas import Article, Categories, RecommendedArticle, UserLikes

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

app.add_middleware(RetryMiddleware)


def vector_to_string(vector) -> str:
    return "[" + ",".join(map(str, vector)) + "]"


async def load_csv_to_db():
    df = pd.read_csv("./data/small_sports_articles.csv")

    async with engine.begin() as conn:
        sql = text(
            "insert into news_articles (news_id, title, general_category, abstract, tf_idf, s_bert, open_ai) values (:news_id, :title, :general_category, :abstract, :tfidf_vector, :sbert_vector, :openai_vector)"
        )
        data = df.to_dict(orient="records")
        await conn.execute(sql, data)

    del df


async def compute_user_profile(
    news_ids: List[str], embedding_model: Recommenders, db: AsyncSession
):
    query = text(f"""
            select {embedding_model.value}
            from news_articles
            where news_id = ANY(:news_ids)
        """)
    result = await db.execute(query, {"news_ids": news_ids})
    embeddings = [row[0] for row in result.fetchall()]

    embeddings_numpy = [
        np.array(json.loads(embedding), dtype=np.float32)
        if isinstance(embedding, str)
        else np.array(embedding, dtype=np.float32)
        for embedding in embeddings
        if embedding
    ]
    user_profile = np.mean(embeddings_numpy, axis=0)
    return vector_to_string(user_profile)


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.tables["news_articles"].drop(
                bind=sync_conn, checkfirst=True
            )
        )
        await conn.execute(text("create extension if not exists vector;"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "create index if not exists tf_idf_hnsw_idx on news_articles using hnsw (tf_idf vector_cosine_ops);"
            )
        )
        await conn.execute(
            text(
                "create index if not exists s_bert_hnsw_idx on news_articles using hnsw (s_bert vector_cosine_ops);"
            )
        )

        await conn.execute(
            text(
                "create index if not exists open_ai_hnsw_idx on news_articles using hnsw (open_ai vector_cosine_ops);"
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


@app.post("/recommend", response_model=List[RecommendedArticle])
async def make_recommendations(
    user_likes: UserLikes, db: AsyncSession = Depends(get_db)
):
    rand_rec = random.sample(list(Recommenders), 2)
    articles = []
    for rec in rand_rec:
        user_profile = await compute_user_profile(user_likes.news_ids, rec, db)
        query = text(f"""
            select 
            news_id,
            title, 
            general_category, 
            abstract 
            from news_articles 
            where news_id != all(:clicked_ids)
            order by {rec.value} <=> :query_vector
            limit 5;
            """)
        result = await db.execute(
            query,
            {"clicked_ids": tuple(user_likes.news_ids), "query_vector": user_profile},
        )
        similar = result.fetchall()
        articles.extend(
            [
                RecommendedArticle(
                    recommender=rec.value,
                    news_id=row.news_id,
                    general_category=row.general_category,
                    title=row.title,
                    abstract=row.abstract,
                )
                for row in similar
            ]
        )
    return articles
