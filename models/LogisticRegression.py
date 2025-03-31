import itertools
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self,seed):
        super().__init__(seed)
    
    def create_model(self):
        return linear_model.LogisticRegression(random_state=self.seed)