import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

if os.path.exists("data")==False:
    os.mkdir("data")

if os.path.exists("model")==False:
    os.mkdir("model")

base_path = os.path.join("names", "yob")
vectorizer = CountVectorizer()
test_size = 0.25
random_state = 6


def create_csv(base_path):
    if os.path.exists("data/data.csv"):
        dataframe = pd.read_csv("data/data.csv")
    else:
        dataframe = pd.DataFrame()
        for year in range(1880, 2021):
            year_data = pd.read_csv(base_path+str(year)+".txt",
                                    header=None,
                                    names=["Name", "Gender", "Count"])
            year_data.insert(1, "Year", year)
            dataframe = pd.concat([dataframe, year_data])
        dataframe.insert(0, "Id", list(range(1, len(dataframe)+1)))
    return dataframe


def prepare_data(dataframe):
    if os.path.exists("data/data.csv") == False:
        dataframe.drop(["Year","Id","Count"], axis = 1, inplace=True)
        dataframe.Gender = dataframe.Gender.map({"F":0,"M":1})
    return dataframe

def split_dataset(dataframe, vectorizer, test_size, random_state):
    X = vectorizer.fit_transform(dataframe.Name.values.astype("U"))
    y = dataframe.Gender
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)
    dataframe.to_csv("data/data.csv", index=False)
    return (X_train, X_test, y_train, y_test)
    