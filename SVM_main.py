import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Oversampling
pot = data[data['Potability']==1]
not_pot = data[data['Potability']==0]

minority = resample(pot, n_samples=1998, replace=True)
data = pd.concat([not_pot, minority])

# Normalization
scaler = preprocessing.MinMaxScaler()
columns = data.columns
x = scaler.fit_transform(data)
data = pd.DataFrame(x, columns=columns)

# Shuffle and split in training and testing sets
data = shuffle(data)
train, test = train_test_split(data, test_size=0.2)

# Split training in samples and labels
train = train.to_numpy()
train_samples, train_labels = train[:, 0:9], train[:, 9].astype(int)

# Split testing in samples and labels
test = test.to_numpy()
test_samples, test_labels = test[:, 0:9], test[:, 9].astype(int)

#model.fit(train_samples, train_labels)
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(train_samples, train_labels)

y_pred=classifier.predict(test_samples)
print(1)
print(y_pred)

ac=classifier.score(test_samples,test_labels)*100
print(2)
print(ac)