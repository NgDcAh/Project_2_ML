import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) # np.mean() tính trung bình


class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000): # lr: learning rate, n_iters: số lần ặp
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # np.zeros tạo mảng toàn giá trị 0
        self.bias = 0

        # gradient descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias # np.dot tính tích vô hướng
            # compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

if __name__ == "__main__":

    data = pd.read_csv("E:\DataSet\score.csv")
    X = data['Hours'].values.reshape(-1, 1)
    y = data['Scores']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    y_pred_line = model.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color="red", s=50)
    m2 = plt.scatter(X_test, y_test, color="black", s=50)
    plt.plot(X, y_pred_line, color="green", linewidth=2, label="Prediction")
    plt.show()