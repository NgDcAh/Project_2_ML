import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2)) # căn ( (x1 - y1) bình + (x2 - y2) bình )
    return distance


class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # tính khoảng cách
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train] # tạo mảng khoảng cách với 1 điểm

        # tìm k điểm gn nhất
        k_indices = np.argsort(distances)[:self.k] # argsort trả về 1 mảng đc sắp xếp thể hiện bằng các chỉ số index
                                                    # [:self.k] lấy k chỉ số đã đc sắp xếp
        k_nearest_labels = [self.y_train[i] for i in k_indices] # trả về mảng là các nhãn ứng với index k đã chọn ra ở trên

        # majority voye
        most_common = Counter(k_nearest_labels).most_common() # trả về danh sách các nhãn và số lần xuất hiện (mảng 2 chiều)
        return most_common[0][0] # lấy ra nhãn dầu tiên


if __name__ == "__main__":

    data = pd.read_csv("E:\DataSet\knn_test.csv")

    X = data.iloc[:,:-1].values
    y = data.iloc[:, 2].values

    from sklearn.preprocessing import LabelEncoder

    # labelEncoder_gender = LabelEncoder()
    # X[:, 0] = labelEncoder_gender.fit_transform(X[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    model = KNN()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)


    # Plot decision boundaries and data points
    plt.figure(figsize=(10, 8))

    labels = ['Class 0', 'Class 1', 'Class 2']

    # Plot training points with labels
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, marker='.', s=50, edgecolors='k',
                label='Training Points')

    # Plot predicted points as stars
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap=plt.cm.Set1, marker='*', s=100, edgecolors='k',
                label='Predicted Points')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('KNN Classification')
    plt.legend()
    plt.show()
