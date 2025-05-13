import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

# 0: 'Nordic walking', 1: 'ascending stairs', 2: 'cycling', 3: 'descending stairs',
# 4: 'ironing', 5: 'lying', 6: 'rope jumping', 7: 'running', 8: 'sitting',
# 9: 'standing', 10: 'transient activities', 11: 'vacuum cleaning', 12: 'walking'
activity_labels = [
    'Nordic walking', 'ascending stairs', 'cycling', 'descending stairs',
    'ironing', 'lying', 'rope jumping', 'running', 'sitting',
    'standing', 'transient activities', 'vacuum cleaning', 'walking'
]

# https://www.kaggle.com/datasets/diegosilvadefrana/fisical-activity-dataset
df = pd.read_csv("clean_physical_activity.csv")

y = df.iloc[:, 1].copy().to_numpy()
X = df.iloc[:, 2:].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Random Forest Classifier that uses the optimal max depth in the range [10, 20]
clf = RandomForestClassifier(oob_score=True, verbose=3)
parameters = {"max_depth": range(10, 20)}
grid_search = GridSearchCV(clf, param_grid=parameters, n_jobs=-1)
grid_search.fit(X_train, y_train)

clf.fit(X_train, y_train)

# Bagging Classifier that finds the optimal number of estimators in the range [1, 7]
bag_clf = BaggingClassifier(estimator=LinearSVC(), oob_score=True, verbose=3)
parameters = { "n_estimators": range(1, 7) }
grid_search_bag = GridSearchCV(bag_clf, param_grid=parameters, n_jobs=-1)
grid_search_bag.fit(X_train, y_train)

bag_clf.fit(X_train, y_train)

# Prints the progress of each
print("Best max_depth:", grid_search.best_params_['max_depth'])
print(f"OOB Score (Random Forest): {clf.oob_score_:.3f}")
print("Best n_estimators:", grid_search_bag.best_params_['n_estimators'])
print(f"OOB Score (Bagging Classifier): {bag_clf.oob_score_:.3f}")

# Graph to show the feature importances in the Random Forest Classifier
importances = pd.DataFrame(clf.feature_importances_, index=df.columns[2:])
importances.plot.bar()
plt.xticks(rotation=45, ha='right')  # Rotate x labels
plt.tight_layout()  # Avoid label cutoff
plt.show()

# Random Forest Classifier confusion matrix
cm = confusion_matrix(y_test, clf.predict(X_test), labels=range(13), normalize="true")
fig, ax = plt.subplots(figsize=(10, 8))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=activity_labels)
disp_cm.plot(ax=ax)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Bagging Classifier confusion matrix
cm = confusion_matrix(y_test, bag_clf.predict(X_test), labels=range(13), normalize="true")
fig, ax = plt.subplots(figsize=(10, 8))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=activity_labels)
disp_cm.plot(ax=ax)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()