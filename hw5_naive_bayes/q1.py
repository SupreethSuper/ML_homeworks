import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Gaussian probability function (given in the homework)
# -------------------------------------------------------------------
def PGauss(mu, sig, x):
    """Compute un-normalised Gaussian probability."""
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0) + 1e-300))


# ===================================================================
#  Naïve Bayes classifier
# ===================================================================
class MyNaiveBayes:
    def __init__(self):
        self.u = None   # means   – shape (n_features, n_classes)
        self.s = None   # stdevs  – shape (n_features, n_classes)
        self.priors = None  # class priors

    def fit(self, X, t):
        """Learn the distribution (mu and stdev) for each feature for each class."""
        self.classes = np.unique(t)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.u = np.zeros((n_features, n_classes))
        self.s = np.zeros((n_features, n_classes))
        self.priors = np.zeros(n_classes)

        for c in self.classes:
            self.priors[c] = np.sum(t == c) / len(t)
            for f in range(n_features):
                self.u[f, c] = X[np.where(t == c), f].mean()
                self.s[f, c] = X[np.where(t == c), f].std()

    def predict(self, X):
        """
        For each observation compute the probability of each class
        (product of probability of each feature given that class
        AND the probability of the class).
        The class with the highest probability is the predicted class.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))

        for c in self.classes:
            # Start with the class prior
            probs[:, c] = self.priors[c]
            for f in range(X.shape[1]):
                probs[:, c] *= PGauss(self.u[f, c], self.s[f, c], X[:, f])

        return np.argmax(probs, axis=1)


# ===================================================================
#  Load data
# ===================================================================
iris = datasets.load_iris()
X = iris.data[:, 0:4]       # features 0, 1, 2, 3
y = iris.target

# -------------------------------------------------------------------
#  Train / test split  (70 / 30)
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ===================================================================
#  Train the model
# ===================================================================
nb = MyNaiveBayes()
nb.fit(X_train, y_train)

# ===================================================================
#  Evaluate – Training set
# ===================================================================
ypred_train = nb.predict(X_train)

print("=" * 60)
print("TRAINING SET RESULTS")
print("=" * 60)
print(f"Number of training samples (observations): {X_train.shape[0]}")
print(f"Number of features used: {X_train.shape[1]}")

miss_train = np.sum(y_train != ypred_train)
acc_train = np.mean(y_train == ypred_train) * 100
print(f"Number of misclassifications: {miss_train}")
print(f"Accuracy: {acc_train:.2f}%")

err_train = np.where(y_train != ypred_train)
print("Errors at indices:", err_train[0])
print("  Actual classification:", y_train[err_train])
print("  Predicted (myNB):     ", ypred_train[err_train])

# ===================================================================
#  Evaluate – Test set
# ===================================================================
ypred_test = nb.predict(X_test)

print()
print("=" * 60)
print("TEST SET RESULTS")
print("=" * 60)
print(f"Number of test samples (observations): {X_test.shape[0]}")
print(f"Number of features used: {X_test.shape[1]}")

miss_test = np.sum(y_test != ypred_test)
acc_test = np.mean(y_test == ypred_test) * 100
print(f"Number of misclassifications: {miss_test}")
print(f"Accuracy: {acc_test:.2f}%")

err_test = np.where(y_test != ypred_test)
print("Errors at indices:", err_test[0])
print("  Actual classification:", y_test[err_test])
print("  Predicted (myNB):     ", ypred_test[err_test])

print()
print(f"Total misclassifications (train + test): {miss_train + miss_test}")
