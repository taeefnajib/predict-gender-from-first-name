import pickle
import numpy as np
import data
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    prog="predgen", description="This program predicts gender based on the first name"
)

parser.add_argument("-n", "--name")
args = parser.parse_args()
fname = args.name

base_path = data.base_path

df = pd.read_csv("data/data.csv")
data.vectorizer.fit_transform(df.Name.values.astype("U"))


model = pickle.load(open("model/model.pkl", "rb"))


def test_name(model, name, vectorizer):
    vector = vectorizer.transform([name]).toarray()
    pred_proba = model.predict_proba(vector)
    if model.predict(vector) == [1]:
        gender = "Male"
    else:
        gender = "Female"
    return {"Gender": gender, "Probability": np.max(pred_proba)}


if __name__ == "__main__":
    print(test_name(model, fname, data.vectorizer))
