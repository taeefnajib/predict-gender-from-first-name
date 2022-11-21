import pickle
import data
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

X_train, X_test, y_train, y_test = data.split_dataset(
    data.prepare_data(data.create_csv(data.base_path)),
    data.vectorizer,
    data.test_size,
    data.random_state,
)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    filename = "model/model.pkl"
    pickle.dump(model, open(filename, "wb"))
    return model


if __name__ == "__main__":
    train_model(model, X_train, y_train)
