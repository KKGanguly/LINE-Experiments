from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from optimizers.base_optimizer import BaseOptimizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

class SuccessiveHalvingOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper,seed):
        super().__init__(config, model_wrapper,seed)

    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging utils not set!!")
        ##new implementation required respecting model_config
        param_names, hyperparameter_space = self.model_wrapper.get_hyperparam_space()
        params = {"classifier__"+param: list(values[:10]) for param, values in zip(param_names, zip(*hyperparameter_space))}
        print(params)
        model = RandomForestClassifier(random_state=42)
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42,k_neighbors=2, sampling_strategy='auto')),
            ('classifier', model)
            ])
        #######################log needs to be called, full thing needs to be fixed###########
        search = HalvingRandomSearchCV(
            pipeline,
            params,
            factor=3,  # The 'eta' or reduction factor
            resource='n_samples',  # Resource to allocate (here, it's the number of trees)
            max_resources=3000,  # Max resources per configuration
            min_resources=200,
            random_state=42,
            scoring = 'roc_auc'
        )
        X_train, y_train, X_test, y_test = self.model_wrapper.get_data()
        self.logging_util.start_logging()
        # Fit the model using the training data
        search.fit(X_train, y_train)
        self.logging_util.stop_logging()
        best_model = search.best_estimator_
        test_accuracy = best_model.score(X_test, y_test)
        print(y_train.value_counts())
        print(test_accuracy)
        print(best_model)

    