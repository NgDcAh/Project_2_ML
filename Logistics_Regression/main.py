import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred


if __name__ == "__main__":

    data = pd.read_csv("E:\DataSet\knn_test.csv")
    data = data[data['color'] != 2]

    X = data.iloc[:, :-1].values
    y = data.iloc[:, 2].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    # Define the range of your feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Generate a mesh grid of points within the feature space
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Flatten and stack the grid points to create input data for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions on the grid points
    predictions = model.predict(grid_points)

    # Reshape the predictions to match the grid shape
    predictions = np.array(predictions).reshape(xx.shape)

    # Plot the training data points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', label='Training Data')

    # Plot the testing data points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Paired, edgecolors='k', marker='s',
                label='Testing Data')

    # Plot the decision boundary
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Paired, alpha=0.8)

    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()