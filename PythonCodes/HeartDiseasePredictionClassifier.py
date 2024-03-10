import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from joblib import dump, load

# Getting a dataset ready

heart_disease = pd.read_csv("../Data/heart-disease.csv")
# print(heart_disease)
X = heart_disease.drop("target", axis=1)
y = heart_disease["target"]
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Preparing a machine learning model

clf = RandomForestClassifier()

# Fitting a model and making predictions

clf.fit(X_train, y_train)
print("Random Classifier score for training data - ", clf.score(X_train, y_train))
print("Random Classifier score for test data - ", clf.score(X_test, y_test))
y_preds = clf.predict(X_test)
print("Random classifier pridiction with actual X test data", y_preds)
print("Actual result - ", y_test.values)

# Experimenting with different classification models

models = {"LinearSVC": LinearSVC(dual="auto"),
          "KNN": KNeighborsClassifier(),
          "SVC": SVC(),
          "LogisticRegression": LogisticRegression(max_iter=10000),
          "RandomForestClassifier": RandomForestClassifier()}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)
print("Different classification model result is - ", results)

# Above-mentioned result if you want to show as a graph

results_df = pd.DataFrame(results.values(),
                          results.keys(),
                          columns=["accuracy"])
results_df.plot.bar()
# plt.show()


# Hyperparameter Tuning , best result is LogisticRegression, so now we need to go ahead with LogisticRegression

LogisticRegression_param_grid = {"C": np.logspace(-4, 4, 20),
                                 "solver": ["liblinear"]}

LogisticRegressionClf = RandomizedSearchCV(estimator=LogisticRegression(),
                                           param_distributions=LogisticRegression_param_grid,
                                           cv=5,
                                           n_iter=5,
                                           verbose=True)

LogisticRegressionClf.fit(X_train, y_train)
print("Best parameter by Hyper Tunning of Logistic Regression is - ", LogisticRegressionClf.best_params_)
print("Logistic Regression score with Hyper Tunning is - ", LogisticRegressionClf.score(X_test, y_test))

LogisticRegressionClf = LogisticRegression(solver="liblinear", C=0.23357214690901212)
LogisticRegressionClf.fit(X_train, y_train)
print("Logistic Regression score with best fitted parameters is -", LogisticRegressionClf.score(X_test, y_test))

# Now it's to import the relative Scikit-Learn methods for each of the classification evaluation metrics we're after.

y_preds = LogisticRegressionClf.predict(X_test)

print("Confusion matrix is - ", confusion_matrix(y_test, y_preds))
print("Classification report is - ", classification_report(y_test, y_preds))
print("Precision score is - ", precision_score(y_test, y_preds))
print("Recall score is - ", recall_score(y_test, y_preds))
print("F1 score is - ", f1_score(y_test, y_preds))

# Whilst this is okay, a more robust way is to calculate them using cross-validation.
cross_val_score1 = cross_val_score(LogisticRegressionClf,
                                   X,
                                   y,
                                   scoring="accuracy",
                                   cv=5)
print("Coross validation score is - ", cross_val_score1)

# EXAMPLE: Taking the mean of the returned values from cross_val_score
# gives a cross-validated version of the scoring metric.
cross_val_acc = np.mean(cross_val_score1)

print("Cross validation accuracy in mean is - ", cross_val_acc)
cross_val_precision = np.mean(cross_val_score(LogisticRegressionClf,
                                              X,
                                              y,
                                              scoring="precision",
                                              cv=5))

print("Cross validation precision in mean is - ", cross_val_precision)
cross_val_recall = np.mean(cross_val_score(LogisticRegressionClf,
                                           X,
                                           y,
                                           scoring="recall",
                                           cv=5))
print("Cross validation recall in mean is - ", cross_val_recall)
cross_val_f1 = np.mean(cross_val_score(LogisticRegressionClf,
                                       X,
                                       y,
                                       scoring="f1",
                                       cv=5))
print("Cross validation f1 in mean is - ", cross_val_f1)

# now export learnt model
dump(LogisticRegressionClf, "trained-LogisticRegression_classifier.joblib")
loaded_clf = load("trained-LogisticRegression_classifier.joblib")

# Evaluate the loaded trained model on the test data
print(loaded_clf.score(X_test, y_test))
