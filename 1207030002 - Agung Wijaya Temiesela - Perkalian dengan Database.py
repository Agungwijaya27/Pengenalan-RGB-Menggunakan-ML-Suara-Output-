import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
#import warnings
from warnings import simplefilter

#Database
FileDB = 'perkalian.txt'
Database = pd.read_csv(FileDB, sep=" ", header = 0)
print("----------------------")
print(Database)

#x = Data, y = Target
x = Database[[u'x']] #ciri1, ciri2, dst
y = Database.Target

regr = LinearRegression().fit(x,y)
regr.score(x, y)

#Data uji
if __name__=='__main__':
    while True:
        print("Prediksi")
        predict = input("Input Prediksi: ")
        predict = np.array([[predict]])
        #ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)
        #Menampilkan data prediksi
        print ("Output = ", regr.predict(predict).astype(int),"\n------------------------")
