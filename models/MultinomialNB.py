from models.base_model import BaseModel
from sklearn.naive_bayes import MultinomialNB

class MultinomialNaiveBayes(BaseModel):
    def __init__(self,seed):
        super().__init__(seed)
    
    def create_model(self):
        return MultinomialNB()