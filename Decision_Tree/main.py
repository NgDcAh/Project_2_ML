import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # dấu * đứng trc value tức là value bắt buộc phải đc truyền dưới dạng đối số từ khóa
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        # nếu self.n_feature chưa đc đặc (none hoặc 0) thì nó sẽ dùng tất cả feature trong X
        # nếu self.n_feature đc gắn giá trị khác 0 thì nó sẽ sử dụng số lượng feature giới hạn của min X.shape[1] và self.n_features
        self.root = self._grow_tree(X, y)

    # xây dựng cây
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape # số mẫu, số feature
        n_labels = len(np.unique(y)) # số nhãn của y

        # kiểm tra điểm dừng
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # nếu độ sâu >= độ sâu max hoặc số nhãn trong nút = 1 hoặc số lượng mẫu < số lượng mẫu tối thiểu có thể tách
            # thì đấy là nút lá
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        #Giảm tính năng: Nếu tính tổng self.n_features, mã giúp giảm số lượng tính năng được xem xét hoặc phân tích.
        # Bằng cách chọn ngẫu nhiên một tập hợp con các chỉ số tính năng, bạn có thể làm việc với một tập hợp các tính năng nhỏ hơn,
        # điều này có thể hữu ích trong trường hợp xử lý toàn bộ các tính năng tốn kém hoặc không cần thiết về mặt tính toán.

        #Tính ngẫu nhiên: Việc lựa chọn ngẫu nhiên đảm bảo rằng các tập hợp con khác nhau của các chỉ số tính năng
        # có thể được tạo ra mỗi khi mã được thực thi. Điều này giới thiệu một yếu tố ngẫu nhiên trong quá trình lựa chọn,
        # cho phép khám phá hoặc đánh giá các kết hợp tính năng khác nhau.

        # tìm cách chia tốt nhất
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # tạo nút con
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh) #X[:, best_feature]: mảng 1 chiều là các giá trị trong best feature
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1) #X[left_idxs, :]: mảng 2 chiều gồm các sample có index ở trong left_idxs
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs: # lặp qua từng feature( cột trong dataset)
            X_column = X[:, feat_idx] # mảng 1 chiều là các giá trị của cột đấy
            thresholds = np.unique(X_column)# trả về 1 mảng chứa các giá trị duy nhất

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten() #argwhere trả về mảng chứa các mảng chứa index của phần tử trong X_column thỏa mãn
        right_idxs = np.argwhere(X_column > split_thresh).flatten() #flatten trả về mảng 1 chiều
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y) # trả về mảng gồm số lần xuất hiện của các phần tử trong y
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y) # tạo đối tượng truy cập vào danh sách của y đếm số lần xuất hiện của các phần tử
        value = counter.most_common(1)[0][0] # lấy ra giá trị phổ biến nhất
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # duyệt cây
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

if __name__ == "__main__":

    data = pd.read_csv("E:\DataSet\IRIS.csv")
    data['flower'] = data['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1})
    data = data.drop('species', axis=1)

    X = data.iloc[:,:-1].values
    y = data.iloc[:, 4].values

    # labelEncoder_gender = LabelEncoder()
    # X[:, 0] = labelEncoder_gender.fit_transform(X[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    model = DecisionTree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)
