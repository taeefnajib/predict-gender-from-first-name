from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd
import numpy as np
from name import Name
import data

model = pickle.load(open("model/model.pkl", "rb"))

app = FastAPI()

df = pd.read_csv("data/data.csv")
data.vectorizer.fit_transform(df.Name.values.astype("U"))


@app.post("/predict/")
def get_prediction(data: Name):
    name = [data.name]
    vector = data.vectorizer.transform(name).toarray()
    gender = model.predict(vector)
    pred_proba = model.predict_proba(vector)
    if gender == [1]:
        return {"Gender": "Male", "Probability": np.max(pred_proba)}
    else:
        return {"Gender": "Female", "Probability": np.max(pred_proba)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
