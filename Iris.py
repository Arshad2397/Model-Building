# Initializing the models# Importing Necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

df = sns.load_dataset("iris")

# Importing Evaluation Metrics
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix

# Initializing all the models
models = {
    "GLM": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC()
}

# Train-Test split
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=28)

# train the models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(xtrain, ytrain)
    y_pred_test = model.predict(xtest)
    y_pred_train = model.predict(xtrain)
    print(f"Testing Report for {name}...")
    print(classification_report(ytest, y_pred_test))
    print(confusion_matrix(ytest, y_pred_test))
    print()
    print(f"Training Report for {name}...")
    print(classification_report(ytrain, y_pred_train))
    print(confusion_matrix(ytrain, y_pred_train))
    print()


# cross-validation of models
cv_results = {}
for name, model in models.items():
    print(f"Cross-validation for {name}...")
    results = cross_val_score(model, xtrain, ytrain, cv=5, scoring='accuracy')
    cv_results[name] = results

cv_results = pd.DataFrame(cv_results)
print(cv_results)

sns.boxplot(data=cv_results)
plt.title("Cross-validation results")
plt.show()

for name,model in models.items():
    plot_confusion_matrix(model, xtrain, ytrain, display_labels=['setosa','versicolor','virginica'])
    plt.title(f"Confusion Matrix for {name}")
    plt.show()