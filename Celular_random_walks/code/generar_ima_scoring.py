# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:51:36 2020

@author: Albert
"""

#generamos las imagenes 2d para el scoring
from cargar_datos import cargar_datos 
from dataset_scoring import create_trajectory_images as cti
import os
import tensorflow as tf

def generar_ima_scoring(path_guardar):
    image_dimension=500
    unoD, dosD, tresD = cargar_datos()
    print(len(dosD))
#    tf_writer = tf.io.TFRecordWriter(os.path.join(path_guardar, 'validation.test_tfrecord'))
#    cti(dosD,label=1, image_dimension=image_dimension,
#                                 tf_writer=tf_writer)

if __name__ == "__main__":
    generar_ima_scoring(r'C:\Users\Albert\Documents\challenge')
    
    
    
    