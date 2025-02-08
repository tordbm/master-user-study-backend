from pydantic import BaseModel
from typing import List

class Categories(BaseModel):
    sports: List[str]
    
class Article(BaseModel):
    news_id: str
    general_category: str
    title: str
    abstract: str
    