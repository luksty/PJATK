import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('Test_classification.csv',sep=';')

df = df.fillna(0)










y = df['class']
X = df[['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

for i in range(1, 11):
    for j in range(1, 11):
        
    

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(i, j), random_state=1)
        
        clf.fit(X_train, y_train)
        
        
        y_hat = clf.predict(X_test)
        print(str(i), " ", str(j))        
        print(accuracy_score(y_test, y_hat))


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(7, 6), random_state=1)

clf.fit(X_train, y_train)


y_hat = clf.predict(X_test)
print(str(i), " ", str(j))        
print(accuracy_score(y_test, y_hat))
        
metrics.roc_auc_score(y_test, y_hat)



