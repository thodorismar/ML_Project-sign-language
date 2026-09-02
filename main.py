#Names/AM: Theodwros Marinos 5426, Georgios Theodwrou 4675

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# 1. LOAD DATA
df = pd.read_csv("sign_mnist_train.csv")

X = df.drop("label", axis=1).values / 255.0
y = df["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Reduce training set for faster debugging
# DEBUG_SIZE = 5000   # how many
# X_train = X_train[:DEBUG_SIZE]
# y_train = y_train[:DEBUG_SIZE]

# 2. EVALUATION FUNCTION
def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds), f1_score(y_val, preds, average="macro")

# 3. EXPERIMENT RUNNER
results = []
# Train a model with the current training set and evaluate it.
# Stores the model name, description, accuracy, F1 score, and the trained model object.
def run_experiment(model, name, desc):
    model.fit(X_train, y_train)  # train the model
    acc, f1 = evaluate(model, X_val, y_val)  # evaluate on validation set
    results.append([name, desc, acc, f1, model])  # save results
    print(f"{name} | {desc} -> acc={acc:.4f}, f1={f1:.4f}")

# 4. kNN
# Dokimazoume diaforetikes times tou k kai diafores metrikes apostasis,
# gia na doume poios sunduasmos dinei kalyteri apodosi sto dataset.
for k in [1, 3, 5, 7, 11]:
    for metric in ["euclidean", "manhattan", "cosine"]:
        run_experiment(
            KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1),
            "kNN",
            f"k={k}, metric={metric}"
        )


# 5. SVM
# Linear SVM: allazoume tin parametro C gia na elegxoume
# poso "skliros" einai o diaxorismos twn klasewn.
for C in [0.1, 1, 10]:
    run_experiment(
        LinearSVC(C=C, dual=False, max_iter=2000, random_state=42),
        "SVM",
        f"linear, C={C}"
    )

# SVM me RBF kernel: dokimazoume diaforetikes times gia C kai gamma,
# gia na doume poso epireazetai i kampylotita tou boundary kai i apodosi.
for C in [0.1, 1, 10]:
    for gamma in ["scale", 0.01]:
        run_experiment(
            SVC(kernel="rbf", C=C, gamma=gamma, cache_size=2000),
            "SVM",
            f"rbf, C={C}, gamma={gamma}"
        )


# 6. Decision Trees
# Dokimazoume diaforetika max_depth, min_samples_leaf kai criterion,
# gia na doume poso epireazoun to overfitting kai tin apodosi tou dentrou.
for depth in [5, 10, 15]:
    for leaf in [1, 5]:
        for crit in ["gini", "entropy"]:
            run_experiment(
                DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                       criterion=crit, random_state=42),
                "DecisionTree",
                f"depth={depth}, leaf={leaf}, crit={crit}"
            )


# 7. Random Forest
# Dokimazoume diaforetiko arithmo dentrwn kai diaforetika max_depth,
# gia na doume poso statheri ginetai i apodosi tou forest.
for n in [50, 100]:
    for depth in [10, 20]:
        run_experiment(
            RandomForestClassifier(n_estimators=n, max_depth=depth,
                                  n_jobs=-1, random_state=42),
            "RandomForest",
            f"trees={n}, depth={depth}"
        )


# 8. AdaBoost
# AdaBoost me aplo base estimator (decision stump).
# Dokimazoume diaforetika n_estimators kai learning_rate.
for n in [50, 100]:
    for lr in [0.5, 1.0]:
        run_experiment(
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n,
                learning_rate=lr,
                random_state=42
            ),
            "AdaBoost",
            f"n={n}, lr={lr}"
        )


# 9. MLP
# Dokimazoume diaforetika hidden layer architectures,
# gia na doume poso voithaei i ayksisi tou vathous tou diktyou.
for layers in [(128,), (128,128), (128,128,128)]:
    run_experiment(
        MLPClassifier(hidden_layer_sizes=layers, activation="relu",
                      max_iter=200, early_stopping=True, random_state=42),
        "MLP",
        f"layers={layers}"
    )


# 10. BEST MODEL SELECTION
df_results = pd.DataFrame(results, columns=["Model", "Desc", "Accuracy", "F1", "Obj"])
# kanoume ignore ta overfitted models
df_filtered = df_results[df_results["F1"] < 0.998]
if df_filtered.empty:
    df_filtered = df_results

best_row = df_filtered.sort_values(["F1", "Accuracy"], ascending=False).iloc[0]
best_model = best_row["Obj"]

print("\nBEST MODEL:")
print(best_row)

# 11. LABEL MAPPING
LABEL_TO_LETTER = {
    0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K',
    10:'L', 11:'M', 12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S',
    18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y'
}

def predict_phrase(model, x_test):
    if x_test.max() > 1:
        x_test = x_test / 255.0
    preds = model.predict(x_test)
    return "".join(LABEL_TO_LETTER[p] for p in preds)
'''
#imports gia ta cnn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

class SimpleCNN(BaseEstimator, ClassifierMixin):
    def __init__(self, num_filters=8, kernel_size=3, pool_size=2, hidden_units=128):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.hidden_units = hidden_units
        self.clf = MLPClassifier(hidden_layer_sizes=(hidden_units,),
                                 activation="relu",
                                 max_iter=200,
                                 early_stopping=True,
                                 random_state=42)

    def _conv2d(self, img, kernel):
        k = self.kernel_size
        out_h = img.shape[0] - k + 1
        out_w = img.shape[1] - k + 1
        out = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                region = img[i:i+k, j:j+k]
                out[i, j] = np.sum(region * kernel)
        return out

    def _relu(self, x):
        return np.maximum(0, x)

    def _maxpool(self, x):
        p = self.pool_size
        out_h = x.shape[0] // p
        out_w = x.shape[1] // p
        out = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                region = x[i*p:(i+1)*p, j*p:(j+1)*p]
                out[i, j] = np.max(region)
        return out

    def _extract_features(self, X):
        X = X.reshape(-1, 28, 28)
        kernels = np.random.randn(self.num_filters, self.kernel_size, self.kernel_size)

        feats = []
        for img in X:
            img_feats = []
            for k in kernels:
                conv = self._conv2d(img, k)
                relu = self._relu(conv)
                pooled = self._maxpool(relu)
                img_feats.extend(pooled.flatten())
            feats.append(img_feats)

        return np.array(feats)

    def fit(self, X, y):
        X_feat = self._extract_features(X)
        self.clf.fit(X_feat, y)
        return self

    def predict(self, X):
        X_feat = self._extract_features(X)
        return self.clf.predict(X_feat)


# 12. TEST AREA 
if __name__ == "__main__":
    print("\n--- TEST: Custom SimpleCNN ---")

    
    # very small test size giati ta for argoun thn python polu
    X_train_small = X_train[:200]
    y_train_small = y_train[:200]

    X_val_small = X_val[:50]
    y_val_small = y_val[:50]

    
    cnn_model = SimpleCNN(num_filters=2, kernel_size=3, pool_size=2, hidden_units=64)

    print("Begin training...")
    cnn_model.fit(X_train_small, y_train_small)

    print("Calculating accuracy...")
    acc, f1 = evaluate(cnn_model, X_val_small, y_val_small)

    print(f"\nResults of SimpleCNN-> acc={acc:.4f}, f1={f1:.4f}")
'''