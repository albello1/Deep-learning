# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:05:34 2020

@author: Albert
"""

#importamos el csv a listas para la task 2
import numpy as np
import os
import csv
import pandas as pd
import pickle

def cargar_datos():
#    with open(r'home\jmgarcia\Albert\RW\task2.txt', newline='') as f:
    with open(r'C:\Users\Albert\Documents\challenge\task2.txt', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
#    print(data[0][0][0])
    #separamos las 3 listas
    unoD = []
    dosD=[]
    tresD=[]
    index_uno=0
    index_dos=0
    index_tres=0
#    for i in range(len(data)):
#        if data[i][0][0]=='1':
#            unoD.insert(i,data[i])
#
#        elif data[i][0][0]=='2':
#            dosD.insert(i,data[i])
#
#        else:
#            tresD.insert(i,data[i])
#    unoD_def=[]
#    dosD_def=[]
#    tresD_def=[]
    
    for j in range(len(data)):
        #empezamos tranformando el elemento de la lista en un array numpy
        lista_actual=data[j][0].split(";")
        lista_float = list(map(float,lista_actual))
        lista_array=np.asarray(lista_float)
        if lista_array[0]==1:
            lista_array = np.delete(lista_array,0)
            unoD.insert(index_uno,lista_array)
            index_uno=index_uno+1
        elif lista_array[0]==2:
            lista_array = np.delete(lista_array,0)
            tam2d = np.shape(lista_array)
            lista_dosD = np.empty([int(tam2d[0]/2),2])
            lista_dosD[:,0]=lista_array[0:int((tam2d[0]/2))]
            lista_dosD[:,1]=lista_array[int((tam2d[0]/2)):]
            dosD.insert(index_dos,lista_dosD)
            index_dos=index_dos+1
        else:
            lista_array = np.delete(lista_array,0)
            tam3d = np.shape(lista_array)
            lista_tresD = np.empty([int(tam3d[0]/3),3])
            lista_tresD[:,0]=lista_array[0:int((tam3d[0]/3))]
            lista_tresD[:,1]=lista_array[int((tam3d[0]/3)):int((tam3d[0]/3)+(tam3d[0]/3))]
            lista_tresD[:,2]=lista_array[int((tam3d[0]/3)+(tam3d[0]/3)):]
            tresD.insert(index_tres,lista_tresD)
            index_tres=index_tres+1
#    with open("test.txt", "wb") as fp:
#        pickle.dump(dosD, fp)
            
    return unoD, dosD, tresD
            
    
        
#    lol = dosD[0][0].split(";")
#    loles=list(map(float, lol))
#    loless = np.asarray(loles)
    
    

    
                

if __name__ == "__main__":
    cargar_datos()