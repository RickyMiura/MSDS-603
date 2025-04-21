from metaflow import FlowSpec, step, Flow, Parameter, JSONType

class ClassifierPredictFlow(FlowSpec):
    vector = Parameter('vector', type=JSONType, required=True)

    @step
    def start(self):
        run = Flow('ClassifierTrainFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
        self.scaler = run['end'].task.data.scaler
        print("Input vector", self.vector)
        self.next(self.end)

    @step
    def end(self):
        scaled_vector = self.scaler.transform([self.vector])
        print('Model', self.model)
        print('Predicted class', self.model.predict(scaled_vector)[0])

if __name__=='__main__':
    ClassifierPredictFlow()