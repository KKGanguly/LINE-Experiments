import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self, seed):
        super().__init__(seed)
        
    def create_model(self):
        return RandomForestClassifier(random_state=self.seed)
        