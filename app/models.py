from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, UUID, Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class StudyResponseModel(Base):
    __tablename__ = "study_response"
    id = Column(UUID, primary_key=True, index=True)
    user_id = Column(UUID, index=True, nullable=False)
    question_id = Column(Integer, index=True, nullable=False)
    response = Column(String, index=True, nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    recommender1 = Column(String, index=True, nullable=True)
    recommender2 = Column(String, index=True, nullable=True)


class NewsArticles(Base):
    __tablename__ = "news_articles"
    news_id = Column(String, primary_key=True, index=True)
    title = Column(String)
    general_category = Column(String)
    abstract = Column(String)
    tf_idf = Column(Vector(373), nullable=False)
    s_bert = Column(Vector(203), nullable=False)
    open_ai = Column(Vector(294), nullable=False)
