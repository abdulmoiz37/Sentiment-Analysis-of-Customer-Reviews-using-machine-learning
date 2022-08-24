#importing of the Modules
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
  
  
main = Tk()
main.title("Sentiment Analysis of Customer Product Reviews Using Machine Learning")
main.geometry("1350x700+0+0")

global filename
global X, Y
global X_train, X_test, y_train, y_test
global tfidf_vectorizer
accuracy = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
global classifier

location_name = ['Arizona', 'Brazil', 'Brooklyn', 'Chennai', 'Florida', 'India', 'Indonesia',
                 'Kerala', 'Kirkwall', 'Pune', 'Sweden', 'United States', 'mexico', 'uk']

Rating_name = ['1','5','1']              

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def Addfile():    
    global filename
    text.delete('1.0', END)
    le = LabelEncoder()
    filename = filedialog.askopenfilename(initialdir="Desktop")
    textdata.clear()
    labels.clear()
    dataset = pd.read_csv(filename)
    print(np.unique(dataset['Review Text']))
    dataset['location'] = pd.Series(le.fit_transform(dataset['location'].astype(str)))
    print(np.unique(dataset['location']))
    for i in range(len(dataset)):
        msg = dataset._get_value(i, 'Review Text')
        label = dataset._get_value(i, 'location')
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(label)
        clean = cleanPost(msg)
        textdata.append(clean)
        text.insert(END,clean+"\n")
        
                     
def preprocessor():
    text.delete('1.0', END)
    global X, Y
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1,2),smooth_idf=False, norm=None, decode_error='replace')
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,str(df))
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    Y = np.asarray(labels)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    print(X)
    print(Y)
    print(Y.shape)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\n\nTotal Reviews found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Total Reviews used to train machine learning algorithms : "+str(len(X_train))+"\n")
    text.insert(END,"Total Reviews used to test machine learning algorithms  : "+str(len(X_test))+"\n")

 

def SVM_algorithm():
    global X, Y
    global tfidf_vectorizer
    global classifier
    global X_train, X_test, y_train, y_test
    global accuracy
    accuracy.clear()
    text.delete('1.0', END) 
    cls = SVC()
    cls.fit(X, Y)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    text.insert(END,"SVM Accuracy : "+str(a)+"\n\n")
    classifier = cls

def Naive_Bayes():
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    text.insert(END,"Naive Bayes Accuracy : "+str(a)+"\n\n")
    classifier = cls



def decesion_tree():
    cls = DecisionTreeClassifier()
    cls.fit(X, Y)
    predict = cls.predict(X_test) 
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    text.insert(END,"Decision Tree Accuracy : "+str(a)+"\n\n")
    classifier = cls

    
def accuracy_graph():
    height = accuracy
    bars = ('Naive Bayes','SVM','Decision Tree')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title('Accuracy Comparison Graph')
    plt.show()

def predict():
    global tfidf_vectorizer
    global classifier
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(testfile)
    text.delete('1.0', END)
    testData = testData.values
    print(testData)
    for i in range(len(testData)):
        msg = testData[i]
        msg1 = testData[i]
        msg = msg[0]
        msg2 = "Review : "
        print(msg)
        review = msg.lower()
        review = review.strip().lower()
        review = cleanPost(review)
        testReview = tfidf_vectorizer.transform([review]).toarray()
        predict = classifier.predict(testReview)[0]
        print(predict)
        text.insert(END,msg2 + str(msg1)+Rating_name[predict]+"\nPositive: "+Rating_name[predict]+"\nNegative: 0\n\n")
        
    
font = ('times', 20, 'bold')
bg_color="#E3CF57" 
title=Label(text="Sentiment Analysis of Customer Product Reviews using Machine Learning",bd=12,relief=GROOVE,bg= bg_color,fg="purple",font=("times new roman",20,"bold"),pady=20).pack(fill=X)     

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

btn_1 = Button(text="Upload Amazon Reviews Dataset",bd=7,command = Addfile)
btn_1.place(x=20,y=150)
btn_1.config(font=ff)

btn_2 = Button(text="Preprocess Dataset",bd=7,command =preprocessor)
btn_2.place(x=20,y=200)
btn_2.config(font=ff)

btn_3 = Button(text="Run SVM Algorithm",bd=7,command = SVM_algorithm)
btn_3.place(x=20,y=250)
btn_3.config(font=ff)

btn_4 = Button(text="Run Naive Bayes Algorithm",bd=7 ,command = Naive_Bayes)
btn_4.place(x=20,y=300)
btn_4.config(font=ff)

btn_5 = Button(text="Run Deceision Tree Algorithm",bd=7,command = decesion_tree)
btn_5.place(x=20,y=350)
btn_5.config(font=ff)

btn_6 = Button(text="Detect Sentiment from Test Reviews",bd=7,command= predict)
btn_6.place(x=20,y=400)
btn_6.config(font=ff)

btn_7 = Button(text="Accuracy Graph",bd=7,command = accuracy_graph)
btn_7.place(x=20,y=450)
btn_7.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=430,y=120,width=1100,height=550)
text.config(font=font1)

main.config(bg='#C0FF3E')
main.mainloop()


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
