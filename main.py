import pickle

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score, r2_score, classification_report

from sklearn.preprocessing import StandardScaler  # Normalize the data
from sklearn.model_selection import train_test_split  # Split the data

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from time import time

# Measure the efficiency of the model
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("fetal_health.csv")
print(df.sample(5))

print(df.info())

df.describe().style.set_properties(**{'background-color': 'grey', 'color': 'white', 'border-color': 'white'})

df.nunique().to_frame(name='Num of unique')

df['fetal_health'].sample(10)

fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(df["baseline value"], alpha=0.5, shade=True, ax=ax, hue=df['fetal_health'], palette="coolwarm")
plt.title('Average Heart Rate Distribution', fontsize=18)
ax.set_xlabel("FHR")
ax.set_ylabel("Frequency")

ax.legend(['Pathological', 'Suspect', 'Normal'])

plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(df["accelerations"], alpha=0.5, shade=True, ax=ax, hue=df['fetal_health'], palette="coolwarm")
plt.title('The Relationship of Acceleration With the Health of the Fetus', fontsize=18)
ax.set_xlabel("Accelerations")
ax.set_ylabel("Frequency")

ax.legend(['Pathological', 'Suspect', 'Normal'])

plt.show()

fig, ax = plt.subplots(figsize=(14, 6))
sns.kdeplot(df["uterine_contractions"], alpha=0.5, shade=True, ax=ax, hue=df['fetal_health'], palette="coolwarm")
plt.title('The Relationship of Uterine Contractions With the Health of the Fetus', fontsize=18)
ax.set_xlabel("Uterine Contractions")
ax.set_ylabel("Frequency")

ax.legend(['Pathological', 'Suspect', 'Normal'])

plt.show()

features = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations',
            'severe_decelerations', 'prolongued_decelerations', 'fetal_health']

sns.pairplot(df[features], hue="fetal_health", height=2.5, palette="coolwarm")
plt.legend(['Pathological', 'Suspect', 'Normal'])
plt.show()

df.hist(figsize=(25, 14))
plt.show()

correlation = df.corr().round(2)
plt.figure(figsize=(14, 7))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()

sns.set_style('white')
sns.set_palette('coolwarm')
plt.figure(figsize=(13, 6))
plt.title('Distribution of correlation of features')
abs(correlation['fetal_health']).sort_values()[:-1].plot.barh()
plt.show()

# Select Features
X = df.drop(columns=['fetal_health'], axis=1)

# Select Target
y = df['fetal_health']

# Set Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2, random_state=44)

print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of testing label:', y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(scaler, open('scaler.pkl', 'wb'))


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='macro')
    rec = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm}


regressors = [
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),
    SVC(),
]

head = 10
for model in regressors[:head]:
    start = time()
    model.fit(X_train, y_train)
    train_time = time() - start
    start = time()
    y_pred = model.predict(X_test)
    predict_time = time() - start
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(y_test, y_pred))
    print("\tMean absolute error:", mean_absolute_error(y_test, y_pred))
    print("\tR2 score:", r2_score(y_test, y_pred))
    print()

svc = SVC()
svc.fit(X_train, y_train)

svc_evaluate = evaluate_model(svc, X_test, y_test)

container = pd.DataFrame(pd.Series(
    {'Accuracy': svc_evaluate['acc'], 'Precision': svc_evaluate['prec'], 'Recall': svc_evaluate['rec'],
     'F1 Score': svc_evaluate['f1']}, name='Result'))
print(container)

sns.heatmap(svc_evaluate['cm'], annot=True, cmap='coolwarm', cbar=False, linewidths=3, linecolor='w',
            xticklabels=['a', 'b', 'c'])
plt.title('Confusion Matrix', fontsize=16)
plt.show()

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

grid_predictions = grid.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))

pickle.dump(grid, open('grid.pkl', 'wb'))

import pandas as pd

from inference import predict

df = pd.read_csv("test.csv")

output = predict(df)

print(output)