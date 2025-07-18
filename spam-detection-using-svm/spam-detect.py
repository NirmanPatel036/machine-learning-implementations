import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import pickle
import cv2 as cv
import tkinter as tk
from tkinter import messagebox, ttk

ps = PorterStemmer()

dset = pd.read_csv('spam.csv')
print(dset)

# apply labels -> 0(not spam) & 1(spam) using Label Encoder, and remove duplicate rows
encoder = LabelEncoder()
dset['Label'] = encoder.fit_transform(dset['Label'])

dset = dset.drop_duplicates(keep='first')
print(dset.head())

# remove any type of punctuation or stop words like the, he, she, etc. Stop words are the words 
# that contribute in the formation of the sentences but these are not useful in detecting whether our SMS is spam or not.

# also utilize the Porter stemming algorithm, which helps reduce words to their base or root form.
# For instance, words like teaching, teach, and taught carry the same underlying meaning.
# The stemmer will convert all of them to a common root form, such as teach.
# This is important because if we skip this step, the CountVectorizer would treat each variation as a separate feature,
# generating multiple columns for essentially the same concept. That could lead to inaccurate interpretations,
# as these words would be viewed as distinct, even though they are semantically related.

def retrieve_importantFeatures(sent):
    sent = sent.lower()
    
    returnList = []
    sent = nltk.word_tokenize(sent)
    for i in sent:
        if i.isalnum():
            returnList.append(i)
    return returnList
 
def remove_stopWords(sent):
    returnList = []
    for i in sent:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            returnList.append(i)
    return returnList
 
def potter_stem(sent):
    returnList = []
    for i in sent:
        returnList.append(ps.stem(i))
    return " ".join(returnList)


# apply the functions to the dataset
dset['imp_feature'] = dset['EmailText'].apply(retrieve_importantFeatures)

dset['imp_feature'] = dset['imp_feature'].apply(remove_stopWords)

dset['imp_feature'] = dset['imp_feature'].apply(potter_stem)

# split the dataset for training and testing
X = dset['imp_feature']
y = dset['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# initiate an SVM model using inbuilt keras function
tfidf = TfidfVectorizer()
feature = tfidf.fit_transform(X_train)
 
tuned_parameters = {'kernel':['linear','rbf'],'gamma':[1e-3,1e-4], 'C':[1,10,100,1000]}
 
model = GridSearchCV(svm.SVC(),tuned_parameters)
model.fit(feature, y_train)


# test the results and save the model
cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)
y_predict = cv.transform(X_test)
print("Accuracy:",model.score(y_predict,y_test))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

"""
def check_spam():
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Input Error", "Please enter a message.")
        return

    input_vector = tfidf.transform([user_input])
    prediction = model.predict(input_vector)[0]

    if prediction == 1:
        result_label.config(text="🚨 This message appears to be SPAM!", fg="#ff4c4c")
    else:
        result_label.config(text="✅ This message appears to be legitimate!", fg="#4caf50")

def clear_input():
    entry.delete(0, tk.END)
    result_label.config(text="")

# --- GUI setup ---
root = tk.Tk()
root.title("📨 Spam Detector")
root.geometry("500x300")
root.resizable(False, False)
root.configure(bg="#121212")

# --- Title ---
title_label = tk.Label(root, text="SMS Spam Classifier", font=("Helvetica", 16, "bold"), bg="#121212", fg="#ffffff")
title_label.pack(pady=(20, 10))

# --- Entry box ---
entry_frame = tk.Frame(root, bg="#121212")
entry_frame.pack(pady=10)

entry = tk.Entry(entry_frame, width=50, font=("Helvetica", 12), bg="#1e1e1e", fg="#ffffff", insertbackground="#ffffff", relief="flat")
entry.pack(ipady=8)

# --- Buttons ---
button_frame = tk.Frame(root, bg="#121212")
button_frame.pack(pady=15)

check_btn = tk.Button(button_frame, text="Check", command=check_spam, font=("Helvetica", 11), bg="black", fg="white", width=12, relief="flat")
check_btn.grid(row=0, column=0, padx=10)

clear_btn = tk.Button(button_frame, text="Clear", command=clear_input, font=("Helvetica", 11), bg="black", fg="white", width=12, relief="flat")
clear_btn.grid(row=0, column=1, padx=10)

# --- Result label ---
result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#121212")
result_label.pack(pady=10)

# --- Run the app ---
root.mainloop()"""