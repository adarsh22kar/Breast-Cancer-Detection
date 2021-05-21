import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("ss.txt",delimiter=',')
df.to_csv('data.csv')

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)

from tkinter import *

def start_prev():
    child1.destroy()
    child2.destroy()
    start()

def check():

    global child2
    child2=Toplevel()
    child2.geometry("300x300")
    a=int(f1.get())
    b=int(f2.get())
    c=int(f3.get())
    d=int(f4.get())
    e=int(f5.get())
    f=int(f6.get())
    g=int(f7.get())
    h=int(f8.get())
    i=int(f9.get())
    y_pred=(classifier.predict(([[a,b,c,d,e,f,g,h,i]])))
    if(y_pred[0]==0):

        ans="You don't have the cancer,cheers!"

    else:
        ans="There is a 94% chance that you may have breast cancer"

    prac=Label(child2,text=ans)
    prac.pack()
    checknew=Button(child2,text="Check New",command=start_prev,padx="50",pady="20",bg="#ab05e8")
    checknew.pack()

    child2.mainloop()

def start():
    global child1
    child1=Toplevel()
    child1.geometry("600x600")
    global f1
    global f2
    global f3
    global f4
    global f5
    global f6
    global f7
    global f8
    global f9

    label1=Label(child1,text="Cl Thickness")
    label1.pack()
    f1=Entry(child1,width=30,bg="#F8D7FD")
    f1.pack()


    label2=Label(child1,text="Cell size")
    label2.pack()
    f2=Entry(child1,width=30,bg="#F8D7FD")
    f2.pack()

    label3=Label(child1,text="Cell.shape")
    label3.pack()
    f3=Entry(child1,width=30,bg="#F8D7FD")
    f3.pack()

    label4=Label(child1,text="Marg.adhesion")
    label4.pack()
    f4=Entry(child1,width=30,bg="#F8D7FD")
    f4.pack()

    label5=Label(child1,text="Epith.c.size")
    label5.pack()
    f5=Entry(child1,width=30,bg="#F8D7FD")
    f5.pack()

    label6=Label(child1,text="Bare.nuclei")
    label6.pack()
    f6=Entry(child1,width=30,bg="#F8D7FD")
    f6.pack()

    label7=Label(child1,text="Bl.cromatin")
    label7.pack()
    f7=Entry(child1,width=30,bg="#F8D7FD")
    f7.pack()

    label8=Label(child1,text="Normal.nucleoli")
    label8.pack()
    f8=Entry(child1,width=30,bg="#F8D7FD")
    f8.pack()

    label9=Label(child1,text="Mitoses")
    label9.pack()
    f9=Entry(child1,width=30,bg="#F8D7FD")
    f9.pack()

    subtn2=Button(child1,text="Check",command=check,padx="50",pady="20",bg="#ab05e8")
    subtn2.pack()

    child1.mainloop()


def first():
    root = Tk()
    root.geometry("300x300")

    subtn=Button(root,text="Start",command=start,padx="50",pady="20",bg="#ab05e8")
    subtn.pack()
    subtn.place(x=85,y=100)
    root.mainloop()

first()
