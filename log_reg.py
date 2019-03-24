import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('Test_classification.csv',sep=';')

df = df.fillna(0)

y = df['class']
X = df[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)


y_hat = clf.predict(X_test)




clf.score(X_test, y_hat )


       
print(accuracy_score(y_test, y_hat))
        
metrics.roc_auc_score(y_test, y_hat)
