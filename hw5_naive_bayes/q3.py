import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# ===================================================================
#  Load data
# ===================================================================
iris = datasets.load_iris()
X = iris.data[:, 0:4]       # features 0, 1, 2, 3
y = iris.target

# -------------------------------------------------------------------
#  Train / test split  (70 / 30)  – same random_state as Q2
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ===================================================================
#  Train sklearn GaussianNB
# ===================================================================
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# ===================================================================
#  Evaluate – Training set
# ===================================================================
ypred_train = gnb.predict(X_train)

print("=" * 60)
print("TRAINING SET RESULTS  (sklearn GaussianNB)")
print("=" * 60)
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of features used:    {X_train.shape[1]}")

miss_train = np.sum(y_train != ypred_train)
acc_train  = np.mean(y_train == ypred_train) * 100
print(f"Number of misclassifications: {miss_train}")
print(f"Accuracy: {acc_train:.2f}%")

err_train = np.where(y_train != ypred_train)
print("errors at indices ", err_train, "actual classificiton ", y_train[err_train],
      " pred sklearn GNB ", ypred_train[err_train])

cm_train = confusion_matrix(y_train, ypred_train, labels=[0, 1, 2])
print("\n number in each class down vs number in each known class across")
print(" confusion matrix ")
print("   0  1  2")
print(cm_train.T)

# ===================================================================
#  Evaluate – Test set
# ===================================================================
ypred_test = gnb.predict(X_test)

print()
print("=" * 60)
print("TEST SET RESULTS  (sklearn GaussianNB)")
print("=" * 60)
print(f"Number of test samples: {X_test.shape[0]}")
print(f"Number of features used: {X_test.shape[1]}")

miss_test = np.sum(y_test != ypred_test)
acc_test  = np.mean(y_test == ypred_test) * 100
print(f"Number of misclassifications: {miss_test}")
print(f"Accuracy: {acc_test:.2f}%")

err_test = np.where(y_test != ypred_test)
print("errors at indices ", err_test, "actual classificiton ", y_test[err_test],
      " pred sklearn GNB ", ypred_test[err_test])

cm_test = confusion_matrix(y_test, ypred_test, labels=[0, 1, 2])
print("\n number in each class down vs number in each known class across")
print(" confusion matrix ")
print("   0  1  2")
print(cm_test.T)

# ===================================================================
#  Comparison with Q2 (custom Naïve Bayes)
# ===================================================================
print()
print("=" * 60)
print("COMPARISON WITH Q2 (custom myNB)")
print("=" * 60)
print(f"Total misclassifications (train + test): {miss_train + miss_test}")
print(f"Training accuracy (sklearn GNB): {acc_train:.2f}%")
print(f"Test accuracy     (sklearn GNB): {acc_test:.2f}%")
print()
print("Comment:")
print("  sklearn GaussianNB uses the full normalised Gaussian PDF")
print("  (including the 1/sqrt(2*pi*sigma^2) normalisation factor),")
print("  while our custom myNB in Q2 uses an un-normalised Gaussian.")
print("  This can lead to slight differences in predicted probabilities,")
print("  but the class predictions are usually very similar because the")
print("  normalisation constant cancels out when comparing classes with")
print("  similar variances.  Any differences in misclassifications are")
print("  due to borderline cases where the normalisation tips the balance.")
