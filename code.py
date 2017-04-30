import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

directory = "creditcard.csv"

data = pd.read_csv(directory, delimiter=',')
data["Scaled Amount"] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data.head()

fraud = data[data.Class != 0]  # Creating a DataFrame with Class value = 1 (Fraud transactions) only
normal = data[data.Class == 0]  # Creating a DataFrame with Class value = 0 (Normal transactions) only

plt.figure(figsize=(10,6))
plt.subplot(121)
fraud[fraud["Amount"]<= 2500].Amount.plot.hist(title="Fraud Transactions")  # Since all the transaction amounts are less than 2500
plt.subplot(122)
normal[normal["Amount"]<=2500].Amount.plot.hist(title="Normal Transactions")  # Since all the transaction amounts are less than 2500 (Clearer view)
plt.show()

data.drop(["Time","Amount"],axis=1,inplace=True)

# Under sampling the data
'''
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = np.array(data[data.Class == 0].index)
rand_normal_indices = np.random.choice(normal_indices, 2508, replace=False)  # Picking random 2508 data to make total data to 3000
rand_normal_indices = np.array(rand_normal_indices)

new_indices = np.concatenate([fraud_indices,rand_normal_indices])  # concatenating the 2 matrices
new_data = data.iloc[new_indices,:]
df2 = pd.DataFrame(new_data)
df = df2.sample(frac=1)
'''
X2 = data.ix[:, data.columns != "Class"]          # Defining the input feature points
Y2 = data.ix[:, data.columns == "Class"]          # Defining the output/target

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.33, random_state=5)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X2_train, Y2_train)
pred = clf.predict(X2_test)

matrix = confusion_matrix(Y2_test, pred)

print("True Positives: ", matrix[1, 1])  # No. of fraud transaction which are predicted fraud
print("True Negatives: ", matrix[0, 0])  # No. of normal transaction which are predicted normal
print("False Positives: ", matrix[0, 1])  # No. of normal transaction which are predicted fraud
print("False Negatives: ", matrix[1, 0])  # No. of fraud Transaction which are predicted normal
print(classification_report(Y2_test, pred))
print "Recall for Random Forest is: ", recall_score(Y2_test, pred)
print "Accuracy for Random Forest is: ", accuracy_score(Y2_test, pred)
print "Precision for Random Forest is: ", precision_score(Y2_test, pred)
print "F1 Score for Random Forest is: ", f1_score(Y2_test, pred)
sns.heatmap(matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Real Class")
plt.show()

important_feat = pd.Series(clf.feature_importances_,index=X2_train.columns).sort_values(ascending=False)
print important_feat  # Printing all the features in descending order of importance

data2 = data[["V11", "V12", "V14", "V16", "V17", "Class"]]  # Taking only first 5 important features
data2.head()
df2 = pd.DataFrame(data2)
X = df2.iloc[:, 0:5]
Y = df2.iloc[:, 5]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=5)
clf2 = SVC()
clf2.fit(X_train, Y_train)
pred2 = clf2.predict(X_test)
matrix2 = confusion_matrix(Y_test, pred2)
print "Number of Fraud cases in training set: ", sum(Y_train)
print "Number of Fraud cases in testing set: ", sum(Y_test)
print "Number of Fraud cases predicted in the testing set: ", sum(pred2)
print "True Positives: ", matrix2[1, 1]  # no of fraud transaction which are predicted fraud
print "True Negatives: ", matrix2[0, 0]  # no. of normal transaction which are predicted normal
print "False Positives: ", matrix2[0, 1]  # no of normal transaction which are predicted fraud
print "False Negatives: ", matrix2[1, 0]  # no of fraud Transaction which are predicted normal
print "Recall for SVM is: ", recall_score(Y_test, pred2)  # Recall for SVM
print "Accuracy for SVM is: ", accuracy_score(Y_test, pred2)  # Accuracy of SVM
print "Precision for SVM is: ", precision_score(Y_test, pred2)  # Precision of SVM
print "F1 Score for SVM is: ", f1_score(Y_test, pred2)  # F1 Score of SVM
print(classification_report(Y_test, pred2))
sns.heatmap(matrix2, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Real Class")
plt.show()
