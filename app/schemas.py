from typing import List
from uuid import UUID

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


class Question(BaseModel):
    question_id: str
    response: str


class UserStudy(BaseModel):
    questionaire: List[Question]


class StudyResponse(BaseModel):
    id: UUID


class UserStudyResponse(BaseModel):
    id: UUID
    user_id: UUID
    question_id: str
    response: str
    timestamp: str

    class Config:
        orm_mode = True
