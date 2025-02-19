from typing import List

from pydantic import BaseModel


class Categories(BaseModel):
    sports: List[str]


class Article(BaseModel):
    news_id: str
    general_category: str
    title: str
    abstract: str


class RecommendedArticle(BaseModel):
    recommender: str
    news_id: str
    general_category: str
    title: str
    abstract: str


class UserLikes(BaseModel):
    news_ids: List[str]
