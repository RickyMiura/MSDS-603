from metaflow import FlowSpec, step, Flow, Parameter, JSONType, resources, timeout, retry, catch, conda_base

@conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.8.1', 'databricks-cli':'0.17.7'}, python='3.9.16')
class ClassifierPredictFlow(FlowSpec):
    vector = Parameter('vector', type=JSONType, required=True)

    @retry(times=2)
    @catch(var='start_error')
    @step
    def start(self):
        run = Flow('ClassifierTrainFlow').latest_run 
        self.train_run_id = run.pathspec 
        self.model = run['end'].task.data.model
        self.scaler = run['end'].task.data.scaler
        print("Input vector", self.vector)
        self.next(self.end)

    @timeout(minutes=5)
    @retry(times=2)
    @catch(var='end_error')
    @step
    def end(self):
        scaled_vector = self.scaler.transform([self.vector])
        print('Model', self.model)
        print('Predicted class', self.model.predict(scaled_vector)[0])

if __name__=='__main__':
    ClassifierPredictFlow()