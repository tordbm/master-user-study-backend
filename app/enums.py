from enum import Enum


class Recommenders(Enum):
    TF_IDF = "tf_idf"
    S_BERT = "s_bert"
    OPEN_AI = "open_ai"


class QuestionaireResponse(Enum):
    TF_IDF = "tf_idf"
    S_BERT = "s_bert"
    OPEN_AI = "open_ai"
    UNSURE = "unsure"
