import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # trả về mảng các phần tử xuất hiện trong y
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64) #trung bình
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) # phương sai
        self._priors = np.zeros(n_classes, dtype=np.float64) # xác suất

        for idx, c in enumerate(self._classes): # duyệt theo index và giá trị
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0) # tính trung bình của X_c theo cột ứng với từng features
            self._var[idx, :] = X_c.var(axis=0) # tnh phương sai của X_c, tính tổng độ lệch bình phương cho mỗi feature / n
            self._priors[idx] = X_c.shape[0] / float(n_samples)# tính xác suất 1 class / tổng class

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx]) # log(P(y))
            posterior = np.sum(np.log(self._pdf(idx, x))) # tổng log P(x|y)
            posterior = posterior + prior
            posteriors.append(posterior) # thêm vào mảng

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]# tìm max

    # hàm mật độ xác suất
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    data = pd.read_csv("E:\DataSet\IRIS.csv")
    data['flower'] = data['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    data = data.drop('species', axis=1)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, 4].values

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))