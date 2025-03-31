from models.base_model import BaseModel
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(BaseModel):
    def __init__(self,seed):
        super().__init__(seed)
    
    def create_model(self):
        return DecisionTreeClassifier(random_state=self.seed)