from metaflow import FlowSpec, step

class ClassifierTrainFlow(FlowSpec):

    @step
    def start(self):
        from sklearn import datasets
        from sklearn.model_selection import train_test_split

        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X,y, test_size=0.2, random_state=0)
        print("Data loaded successfully")
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data)
        self.train_data = self.scaler.transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)
        self.next(self.train_knn, self.train_svm)

    @step
    def train_knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }

        self.model = KNeighborsClassifier()
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.train_data, self.train_labels)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.next(self.choose_model)

    @step
    def train_svm(self):
        from sklearn import svm

        self.model = svm.SVC(kernel='poly')
        self.model.fit(self.train_data, self.train_labels)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        import mlflow
        mlflow.set_tracking_uri('https://mlflow-test-run-803841580416.us-west2.run.app')
        mlflow.set_experiment('metaflow-experiment')

        def score(inp):
            return inp.model, inp.model.score(inp.test_data, inp.test_labels), getattr(inp, "best_params", None)

        self.results = sorted(map(score, inputs), key=lambda x: -x[1])
        self.model = self.results[0][0]
        best_params = self.results[0][2]

        self.scaler = inputs[0].scaler

        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, artifact_path = 'metaflow_train', registered_model_name="metaflow-wine-model")
            if best_params is not None:
                mlflow.log_params(best_params)
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        for model, score, best_params in self.results:
            print(f"{type(model).__name__}: {score:.4f}")
            if best_params:
                print(f"  Best Params: {best_params}")
        print('\nSelected Model:')
        print(self.model)
        self.scaler = self.scaler


if __name__=='__main__':
    ClassifierTrainFlow()