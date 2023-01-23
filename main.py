import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing


# Read .csv file
data = pd.read_csv('water_potability.csv')

# Pairplot
sn.pairplot(data, hue='Potability', height=1.7)
plt.show()

# Correlation maxtrix
plt.figure(figsize=(15, 8))
sn.heatmap(data.corr(), annot=True).set(title='Correlation matrix')
plt.show()

# Samples' potability
sn.countplot(x=data['Potability'])
print(data['Potability'].value_counts())

# Check where are unknown values and replace with mean
print(data.isnull().sum())
data['ph'] = data['ph'].replace(np.NaN, data['ph'].mean())
data['Sulfate'] = data['Sulfate'].replace(np.NaN, data['Sulfate'].mean())
data['Trihalomethanes'] = data['Trihalomethanes'].replace(np.NaN, data['Trihalomethanes'].mean())

# Normalization
scaler = preprocessing.MinMaxScaler()
columns = data.columns
x = scaler.fit_transform(data)
data = pd.DataFrame(x, columns=columns)

# Shuffle and split in training and testing sets
data = shuffle(data)
train, test = train_test_split(data, test_size=0.2)

# Oversampling
pot = train[train['Potability']==1]
not_pot = train[train['Potability']==0]

minority = resample(pot, n_samples=len(not_pot), replace=True)
train = pd.concat([not_pot, minority])

# Split training in samples and labels
train = train.to_numpy()
train_samples, train_labels = train[:, 0:9], train[:, 9].astype(int)

# Split testing in samples and labels
test = test.to_numpy()
test_samples, test_labels = test[:, 0:9], test[:, 9].astype(int)

acc = []
for k in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
    model.fit(train_samples, train_labels)
    preds = model.predict(test_samples)
    acc.append(accuracy_score(test_labels, preds))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), acc, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=8)
plt.title('Minkowski Distance')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

print(acc.index(max(acc)))
model = KNeighborsClassifier(n_neighbors=acc.index(max(acc)) + 1, metric='minkowski')
model.fit(train_samples, train_labels)
preds = model.predict(test_samples)
conf_matrix = confusion_matrix(test_labels, preds)

sn.heatmap(conf_matrix, annot=True, fmt='g')
plt.show()
print(classification_report(test_labels, preds))
