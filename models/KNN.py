import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModel

class KNN(BaseModel):
    def __init__(self,seed):
        super().__init__(seed)
    
    def create_model(self):
        return KNeighborsClassifier()