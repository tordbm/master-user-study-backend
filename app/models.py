from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StudyResponseModel(Base):
    __tablename__ = "study_response"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(String, index=True)


class NewsArticles(Base):
    __tablename__ = "news_articles"
    news_id = Column(String, primary_key=True, index=True)
    title = Column(String)
    general_category = Column(String)
    abstract = Column(String)
    tf_idf = Column(Vector(10), nullable=False)
    s_bert = Column(Vector(10), nullable=False)
    open_ai = Column(Vector(10), nullable=False)
