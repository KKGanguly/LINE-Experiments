import itertools
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from models.base_model import BaseModel


class SVM(BaseModel):
    def __init__(self,seed):
        super().__init__(seed)
        
    def create_model(self):
        return svm.SVC(random_state=self.seed, probability=False)