from sklearn import svm

train_data = [[1, 1], [2, 1], [1, 2], [3, 2], [4, 2], [5, 1], [3, 3]]

train_label = [1, 1, 1, -1, -1, -1, -1]

svm_clf = svm.LinearSVC()

svm_clf.fit(train_data, train_label)

print("test [1, 3] and [5, 3]")
print(svm_clf.predict([[1, 3], [5, 3]]))

"""
print("get support vectors")
print(svm_clf.support_vectors_)

print("get number of support vectors for each class")
print(svm_clf.n_support_)
"""
