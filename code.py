import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
df=pd.read_csv('C:/Users/Mallika/Desktop/news.csv',low_memory=False)
#Get shape and head
df.shape
df.head()
labels=df.label
labels.head()
x_train,x_test,y_train,y_test=train_test_split(df['text'].values.astype('U'), labels, test_size=0.2, random_state=7)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
pac=PassiveAggressiveClassifier(max_iter=50)
y_train.fillna(method='ffill',inplace=True)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test.astype(str),y_pred.astype(str))
print(f'Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test.astype(str),y_pred.astype(str), labels=['FAKE','REAL'])
from tkinter import *
from tkinter import Tk
top=Tk()
top.geometry("600x200")
def button_command():
    text = entry.get()
    input_data=[text]
    vectorized_input_data=tfidf_vectorizer.transform(input_data)
    prediction=pac.predict(vectorized_input_data)
    li = Listbox(top, height=6, width=15, font=("arial", "20"))
    li.insert(1,prediction)
    li.pack()
entry=Entry(top,width=50,font=("arial","10"))
entry.pack()
Button(top,text="Predict",command=button_command).pack()
top.mainloop()
