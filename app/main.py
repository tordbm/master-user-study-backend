import json
import random
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import engine
from app.enums import Recommenders
from app.models import Base, StudyResponseModel
from app.schemas import (
    Article,
    Categories,
    RecommendedArticle,
    StudyResponse,
    UserLikes,
    UserStudy,
    UserStudyResponse,
)

from .database import get_db


@asynccontextmanager
async def lifespan(app):
    async with engine.begin() as conn:
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
        result = await conn.execute(text("select exists (select 1 from news_articles)"))
        has_entries = result.scalar()

    # await load_questions()
    if not has_entries:
        await load_csv_to_db()
    else:
        print("Data already present, skipping data loading...")

    yield


app = FastAPI(lifespan=lifespan)

origins = [
    "https://master-user-study.vercel.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def vector_to_string(vector) -> str:
    return "[" + ",".join(np.char.mod("%f", vector)) + "]"


async def load_csv_to_db():
    df = pd.read_csv("./data/balanced_small_articles.csv")

    async with engine.begin() as conn:
        sql = text(
            "insert into news_articles (news_id, title, general_category, abstract, tf_idf, s_bert, open_ai) values (:news_id, :title, :general_category, :abstract, :tfidf_vector, :sbert_vector, :openai_vector)"
        )
        data = df.to_dict(orient="records")
        await conn.execute(sql, data)

    del df


async def load_questions():
    questions = []

    with open("./data/questionaire.txt", "r") as file:
        for index, value in enumerate(file.readlines()):
            questions.append({"question_id": index + 1, "question": value.strip()})
    async with engine.begin() as conn:
        sql = text(
            "insert into questions (question_id, question) values (:question_id, :question)"
        )
        await conn.execute(sql, questions)


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

    if not embeddings_numpy:
        raise HTTPException(
            status_code=400, detail="No embeddings found for user profile"
        )

    user_profile = np.mean(embeddings_numpy, axis=0)
    return vector_to_string(user_profile)


@app.post("/articles", response_model=List[Article])
async def fetch_sports_articles(
    categories: Categories,
    shown_articles: List[str] = [],
    db: AsyncSession = Depends(get_db),
):
    try:
        placeholders = ", ".join(
            [f":category{i}" for i in range(1, len(categories.sports) + 1)]
        )
        query = text(
            f"""
            select
            news_id,
            title,
            general_category,
            abstract
            from news_articles
            where (
                general_category in ({placeholders}) and
                news_id != all(:articles_shown)
                )
            """
        )

        sports = {
            f"category{i + 1}": categories.sports[i]
            for i in range(len(categories.sports))
        }

        result = await db.execute(
            query,
            {
                **sports,
                "articles_shown": tuple(shown_articles),
            },
        )
        articles = result.fetchall()

        if not articles:
            raise HTTPException(
                status_code=404, detail="No articles found for the given categories"
            )

        if len(articles) < 9:
            raise HTTPException(
                status_code=404, detail="Not enough articles in dataset"
            )

        articles_dict = [
            {
                "news_id": article[0],
                "title": article[1],
                "general_category": article[2],
                "abstract": article[3],
            }
            for article in articles
        ]

        sampled_articles = []
        for category in categories.sports:
            category_articles = [
                article
                for article in articles_dict
                if article["general_category"] == category
            ]
            sampled_articles.extend(
                random.sample(category_articles, min(3, len(category_articles)))
            )

        shuffled_articles = [Article(**article) for article in sampled_articles]
        random.shuffle(shuffled_articles)

        return shuffled_articles

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching articles: {str(e)}"
        )


@app.post("/recommend", response_model=List[RecommendedArticle])
async def make_recommendations(
    user_likes: UserLikes, db: AsyncSession = Depends(get_db)
):
    try:
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
                {
                    "clicked_ids": tuple(user_likes.news_ids),
                    "query_vector": user_profile,
                },
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

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making recommendations: {str(e)}"
        )


@app.post("/insert_study_response", response_model=StudyResponse)
async def insert_user_response(
    user_study: UserStudy, db: AsyncSession = Depends(get_db)
):
    try:
        user_id = uuid4()

        study_responses = []

        for item in user_study.questionaire:
            study_response = StudyResponseModel(
                id=uuid4(),
                user_id=user_id,
                question_id=item.question_id,
                response=item.response,
                timestamp=datetime.now(),
                recommender1=item.recommender1,
                recommender2=item.recommender2,
            )
            study_responses.append(study_response)

        db.add_all(study_responses)

        await db.commit()

        return {"id": user_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not add response: {str(e)}")


@app.get("/all_study_responses", response_model=List[UserStudyResponse])
async def get_all_responses(db: AsyncSession = Depends(get_db)):
    query = text("""
                 select * from study_response
                 order by timestamp desc
                 """)

    result = await db.execute(query)

    res = result.fetchall()

    if not res:
        return []

    study_responses = [
        UserStudyResponse(
            id=row.id,
            user_id=row.user_id,
            question_id=row.question_id,
            response=row.response,
            timestamp=str(row.timestamp),
            recommender1=row.recommender1,
            recommender2=row.recommender2,
        )
        for row in res
    ]

    return study_responses


@app.get("/participants")
async def get_participants(db: AsyncSession = Depends(get_db)):
    query = text("""
                 select count(distinct user_id) from study_response
                 """)

    result = await db.execute(query)

    res = result.scalar()

    return {"participants": res}


@app.get("/stats_per_answer")
async def get_stats_per_answer(db: AsyncSession = Depends(get_db)):
    query = text("""
                 SELECT 
                    q.question_id, 
                    q.question, 
                    sr.response, 
                    COUNT(sr.response) AS answer_count
                FROM study_response sr
                JOIN questions q ON q.question_id = sr.question_id
                WHERE sr.response IN ('open_ai', 's_bert', 'tf_idf', 'list1', 'list2', 'unsure')
                GROUP BY q.question_id, q.question, sr.response
                ORDER BY q.question_id, sr.response
                 """)

    result = await db.execute(query)

    res = result.fetchall()

    stats = {}
    for row in res:
        question_id, question, response, answer_count = row
        if question_id not in stats:
            stats[question_id] = {"question": question, "responses": {}}
        stats[question_id]["responses"][response] = answer_count

    return stats
