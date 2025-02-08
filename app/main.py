from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import random
import pandas as pd
from app.models import Base
from app.database import engine

from app.schemas import Article, Categories

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
def fetch_sports_articles(categories: Categories):
    print(categories)
    df = pd.read_csv("sports_articles.csv")

    required_columns = {"news_id", "general_category", "title", "abstract"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=500, detail="CSV file missing required columns")
    
    df["abstract"] = df["abstract"].fillna("No abstract available")
    
    filtered_df = df[df["general_category"].isin(categories.sports)]

    articles = []
    for category in categories.sports:
        category_articles = filtered_df[filtered_df["general_category"] == category]
        sampled_articles = category_articles.sample(n=min(3, len(category_articles)), random_state=random.randint(1, 10000))
        articles.extend(sampled_articles.to_dict(orient="records"))
    return [Article(**article) for article in articles]