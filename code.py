import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

directory = "creditcard.csv"

data = pd.read_csv(directory, delimiter=',')
df = pd.DataFrame(data)

a = df[df.Class != 0]           # Creating a DataFrame with Class value = 1 only
b = df[df.Class == 0]           # Creating a DataFrame with Class value = 0 only
c = [a, b]
e = pd.concat(c)            # Concatinating the 2 matrices a and b
f = e.sample(frac=1)        # Shuffling the final DataFrame

X = f.iloc[:, 1:29]          # Defining the input feature points
target = f.Class
Y = target                  # Defining the output/target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

clf = SVC(kernel='rbf', C=1, degree=3, max_iter=100)
clf.fit(X_train, Y_train)
pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

print "Predicted Values: ", pred
print "R-Squared Score: ", score

print "No. of frauds in training: ", sum(Y_train)
print "No. of frauds in testing: ", sum(Y_test)
print "No. of frauds in predicting: ", sum(pred)

recall = recall_score(Y_test, pred, average='macro')
precision = precision_score(Y_test, pred, average='macro')
f1 = f1_score(Y_test, pred, average='macro')

print "Recall of our Algorithm: ", recall
print "Precision of our Algorithm: ", precision
print "F1 Score of our Algorithm: ", f1
