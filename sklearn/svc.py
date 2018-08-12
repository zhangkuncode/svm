from sklearn import svm

train_data = [[1, 1], [2, 1], [1, 2], [3, 2], [4, 2], [5, 1], [3, 3]]

train_label = [1, 1, 1, -1, -1, -1, -1]

"""
class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, 
		              coef0=0.0, shrinking=True, probability=False, 
					  tol=0.001, cache_size=200, class_weight=None, 
					  verbose=False, max_iter=-1, decision_function_shape=’ovr’, 
					  random_state=None)
"""
# rbf guassian kernel
#svm_clf = svm.SVC()

# linear kernel
svm_clf = svm.SVC(1, "linear")

svm_clf.fit(train_data, train_label)

print("test [1, 3] and [5, 3]")
print(svm_clf.predict([[1, 3], [5, 3]]))

print("get support vectors")
print(svm_clf.support_vectors_)

print("get number of support vectors for each class")
print(svm_clf.n_support_)

