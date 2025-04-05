import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

data = pd.read_csv('data/heart_attack_prediction_indonesia.csv')

data_y = data.iloc[:,-1]
data = data.iloc[:, :-1]
data_y = data_y.values.reshape(-1,1)

numerical_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_trainsformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, make_column_selector(dtype_include = ['int', 'float'])),
        ("cat", categorical_trainsformer, make_column_selector(dtype_exclude = ['int', 'float']))
    ]
)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ]
)

clf.fit(data, data_y)
data_new = clf.transform(data)
data_new = pd.DataFrame(data_new)
data_new['heart_attack'] = data_y
data_new.to_csv('data/processed_heart_attack_prediction.csv')

with open('data/pipeline.pkl', 'wb') as f:
    pickle.dump(clf, f)