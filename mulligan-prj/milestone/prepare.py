x = X/255.0
print(np.max(x))

lbls = np.zeros((X.shape[0],120)) # we have 120 dog breeds

for i in range(X.shape[0]):
  #print(i)
  lbls[i][labels[i]-1] = 1
  
# split dataset into training and validation (test set)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(x, labels)

for train_index, test_index in skf.split(X, labels):
  X_train, X_test =    x[train_index],    x[test_index]
  y_train, y_test = lbls[train_index], lbls[test_index]

print("X_train:")
print(X_train.shape)
print("X_test")
print(X_test.shape)
print("y_train:")
print(y_train.shape)
print("y_test")
print(y_test.shape)
  
#print(labels[0])
#print(lbls[0])

# X = x
# labels = lbls