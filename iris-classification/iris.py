import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

#make a directory for saving plots
os.makedirs("plots", exist_ok=True)

#load the dataset
df = pd.read_csv("Iris.csv")
print(df.head())

#delete unwanted column
df = df.drop(columns = ['Id'])
print(df.head())

#display stats of the dataset
print(df.describe())
print(df.info())
print(df["Species"].value_counts())

#preprocessing the dataset
print(df.isnull().sum()) #checks for any null values

#exploratory data analysis
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
plt.figure(figsize=(10, 6))

for i, feature in enumerate(features):
    plt.subplot(2, 2, i + 1)
    df[feature].hist(color='skyblue', edgecolor='black')
    plt.title(feature)

plt.tight_layout()
plt.show()

#scatterplot for sepals
colours = ["red", "orange", "blue"]
species = ["Iris-virginica", "Iris-versicolor", "Iris-setosa"]
for i in range(3):
	x = df[df["Species"] == species[i]]
	plt.scatter(x["SepalLengthCm"], x["SepalWidthCm"], c = colours[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.savefig("plots/sepals.png")
plt.show()

#scatterplot for petals
for i in range(3):
	x = df[df["Species"] == species[i]]
	plt.scatter(x["PetalLengthCm"], x["PetalWidthCm"], c = colours[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()
plt.savefig("plots/petals.png")
plt.show()

#find the correlation matrix
corr = df.select_dtypes(include='number').corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, ax=ax)
plt.title("Feature Correlation Matrix")
plt.show()

#label encoding
'''
In machine learning, we usually deal with datasets which contains multiple labels in one or more than one columns. These
labels can be in the form of words or numbers. Label Encoding refers to converting the labels into numeric form so as to
convert it into the machine-readable form'''
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])
print(df.head()) #to check whether the labels are converted to numeric values or not

#model training: train - 70; test - 30;
X = df.drop(columns = ["Species"])
Y = df["Species"]
x_train , x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#evaluation metrics for performance check for different models
#for LR
model = LogisticRegression()
model.fit(x_train, y_train)
print("LR Accuracy: ", model.score(x_test, y_test)*100, "%")

model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("KNN Accuracy: ", model.score(x_test, y_test)*100, "%")

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Decision Tree Accuracy: ", model.score(x_test, y_test)*100, "%")

# save the model
filename = 'saved_model.sav'
joblib.dump(model, open(filename, 'wb'))

#test the model
load_model = joblib.load(open(filename,'rb'))
pred = model.predict(pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=x_train.columns))

if pred == 0:
	print("Iris-setosa")
elif pred == 1:
	print("Iris-versicolor")
else:
	print("Iris-virginica")