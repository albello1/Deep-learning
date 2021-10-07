# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 01:56:05 2020

@author: Albert
"""


import os
import glob
import numpy as np
import tensorflow as tf
from andi2 import ANDI
from skimage.draw import line
import math

SEED = 1212
n = 5
lista_vars = []
lista_esperas = []


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_model_dataset(model, alpha_start, alpha_stop, alpha_step=0.1, N=200, mu=0.5, sigma=0, traj_l_start=200, traj_l_stop=800, traj_step=100,
#def create_model_dataset(model, alpha_start, alpha_stop, alpha_step=0.1, N=1, mu=0.5, sigma=0, traj_l_start=200, traj_l_stop=800, traj_step=100,
                         dimension=2, output_directory=None):
    #vamos a trabajar con trayectories de diferente length, para ello podemos emplear listas que se pueden rellenar con
    #matrices de longitud variable
    if model == 0:
        model_name = 'ctrw'
    elif model == 1:
        model_name = 'fbm'
    elif model == 2:
        model_name = 'lw'
    elif model == 3:
        model_name = 'attm'
    elif model == 4:
        model_name = 'sbm'
    else:
        raise ValueError('Unexpected model type: ' + str(model))

    print('Creating trajectories for model {}...'.format(model_name))

    andi = ANDI()
    exponents = np.arange(alpha_start, alpha_stop, alpha_step)
    longs = np.arange(traj_l_start, traj_l_stop, traj_step)
    num_exponents = len(exponents)
    num_exponents = num_exponents*N
    #creamos la lista y la vamos rellenando con los diferentes casos
    trajectories_list=[]
    for i in range(longs.shape[0]):
        dataset = andi.create_noisy_localization_dataset(T=longs[i], N=N, mu=mu, sigma=sigma,
                                                     exponents=exponents, models=model, dimension=dimension)
        
   
        trajectories = np.moveaxis(np.reshape(dataset[:, 2:], [num_exponents, longs[i], 2], order='F'), 0, 2)
        trajectories_list.append(trajectories)

        
#    dataset = andi.create_noisy_localization_dataset(T=trajectory_length, N=N, mu=mu, sigma=sigma,
#                                                     exponents=exponents, models=model, dimension=dimension)
#    trajectories = np.moveaxis(np.reshape(dataset[:, 2:], [num_exponents, trajectory_length, 2], order='F'), 0, 2)

    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        np.save(os.path.join(output_directory, model_name + '_trajectories'), trajectories)
        np.save(os.path.join(output_directory, model_name + '_labels'), model)

    return trajectories_list, model, model_name


def create_trajectory_images(trajectories_list, label, image_dimension=500, tf_writer=None, output_directory=None):
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    print('Creating images for model {}...'.format(label))
    for j in range(len(trajectories_list)):
        trajectories = trajectories_list[j]

        N = trajectories.shape[-1]
        T = trajectories.shape[0]
        scaling_factor = image_dimension - 1
        for i in range(N):
            X = trajectories[:, 0, i]
            Y = trajectories[:, 1, i]
#            X = trajectories[-150:, 0, i]
#            Y = trajectories[-150:, 1, i]
            min_x = np.min(X)
            max_x = np.max(X)
            min_y = np.min(Y)
            max_y = np.max(Y)
            x_range = max_x - min_x
            y_range = max_y - min_y
            if x_range == 0:
                x_range = 2 * min_x
                min_x = 0
            if y_range == 0:
                y_range = 2 * min_y
                min_y = 0

            image = np.zeros([image_dimension, image_dimension, 2])
#            image = np.zeros([image_dimension, image_dimension, 1])
            dist_acu = 0
            lista_x=[]
            lista_y=[]
            lista_max = []
            lista_var =[]
            lista_espera =[]
            for t in range(0, T - 1):
#            for t in range(0, 150 - 1):
                x0 = X[t]
                y0 = Y[t]
                x1 = X[t + 1]
                y1 = Y[t + 1]

                x_start = int(round((scaling_factor * (x0 - min_x)) / x_range))
                y_start = int(round((scaling_factor * (y0 - min_y)) / y_range))
                x_end = int(round((scaling_factor * (x1 - min_x)) / x_range))
                y_end = int(round((scaling_factor * (y1 - min_y)) / y_range))
                xx, yy = line(x_start, y_start, x_end, y_end)
                dist_acu += np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
                
                image[xx, yy, 0] = (t + 1)/T
#                image[xx, yy, 0] = t + 1
#                image[xx, yy, 0] = dist_acu
                image[xx, yy, 1] = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
#                print(np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))
                lista_x.append(x_start)
                lista_y.append(y_start)
            
            maxima_espera= calculo_max_espera(lista_x,lista_y,3)
            
            if maxima_espera > 200:
                maxima_espera = 1
            else:
                maxima_espera = maxima_espera/200
                
            
            image[:,:,0]=image[:,:,0]+maxima_espera
            
            image[:,:,0]=np.where(image[:,:,0] == maxima_espera, image[:,:,0], image[:,:,0]=0)

        
            
                
            if output_directory is not None:
                np.save(os.path.join(output_directory, str(i)), image)
            if tf_writer is not None:
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': _bytes_feature(image.tobytes()),
                    'height': _int64_feature(image_dimension),
                    'width': _int64_feature(image_dimension),
                    'depth': _int64_feature(2),
#                    'depth': _int64_feature(1),
                    'label': _int64_feature(label)
                    }))
                tf_writer.write(example.SerializeToString())
        
    

def create_datasets(output_directory, image_dimension=500, is_training=True):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create datasets
    ctrw = create_model_dataset(model=0, alpha_start=0.5, alpha_stop=1)
    fbm = create_model_dataset(model=1, alpha_start=0.5, alpha_stop=1)
    lw = create_model_dataset(model=2, alpha_start=1, alpha_stop=1.5)
    attm = create_model_dataset(model=3, alpha_start=1.1, alpha_stop=1.6)
    sbm = create_model_dataset(model=4, alpha_start=1, alpha_stop=1.5)

    if not is_training:
        tf_writer = tf.io.TFRecordWriter(os.path.join(output_directory, 'validation.test_tfrecord'))
    # Create trajectory images and store them in TFRecords
    for model in (ctrw, fbm, lw, attm, sbm):
        if is_training:
            file_name = '{}.train_tfrecord'.format(model[2])
            tf_writer = tf.io.TFRecordWriter(os.path.join(output_directory, file_name))
        create_trajectory_images(trajectories_list=model[0], label=model[1], image_dimension=image_dimension,
                                 tf_writer=tf_writer)
        if is_training:
            tf_writer.close()
    if not is_training:
        tf_writer.close()


def train_dataset(tfrecords_directory, batch_size, shuffle_size=128, repeat=True):
    tfrecords_paths = glob.glob(os.path.join(tfrecords_directory, '*.train_tfrecord'))
    if not tfrecords_paths:
        raise ValueError('Empty tfrecord_paths: {}'.format(tfrecords_paths))
    weights = [1.0 / len(tfrecords_paths)] * len(tfrecords_paths)
    datasets = []
    for tfrecord_path, weight in zip(tfrecords_paths, weights):
        if not os.path.exists(tfrecord_path):
            raise FileNotFoundError('Dataset {} not found'.format(tfrecord_path))

        print('Adding dataset for train: {}  --  Weight: {}'.format(tfrecord_path, weight))
        dataset = tf.data.TFRecordDataset(filenames=tfrecord_path)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.shuffle(buffer_size=shuffle_size, seed=SEED)
        if repeat:
            dataset = dataset.repeat()
        datasets.append(dataset)

    if len(datasets) > 1:
        dataset = tf.data.experimental.sample_from_datasets(datasets=datasets, weights=weights)
    else:
        dataset = datasets[0]
    dataset = dataset.map(map_func=parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def validation_dataset(tfrecords_directory, batch_size, repeat=True):
    tfrecord_path = glob.glob(os.path.join(tfrecords_directory, '*.test_tfrecord'))
    if not tfrecord_path:
        raise FileNotFoundError('Validation dataset {} not found!'.format(tfrecord_path))
    tfrecord_path = tfrecord_path[0]

    print('Adding dataset for validation: {}'.format(tfrecord_path))
    dataset = tf.data.TFRecordDataset(filenames=tfrecord_path)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(map_func=parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if batch_size > 0:
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


@tf.function
def parse_tfrecord(serialized):
    features = {
        'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
        'height': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        'width': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        'depth': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }

    example = tf.io.parse_single_example(serialized=serialized, features=features)

    image = tf.cast(x=tf.io.decode_raw(input_bytes=example['image'], out_type=tf.as_dtype(np.float64)),
                    dtype=tf.float32)

    image = tf.reshape(tensor=image, shape=(example['width'], example['height'], example['depth']))
    label = example['label']
#    max_espera = example['max_espera']



    return image, label
#    return (image, max_espera), label

def calculo_max_espera(listax, listay, ventana):
    esperas=[]
    tam_list = len(listax)
#    listax = np.array(listax)
#    listay= np.array(listay)
    for i in range(tam_list-1):
        index=0
        for j in range(i,tam_list-1):
#            print(listax[j+1+i])
#            print(listax[i])
#            print(listay[j+1+i])
#            print(listay[i])
#            if j==tam_list:
#                break
#            if i==tam_list:
#                break
            if np.sqrt((listax[j+1] - listax[i]) ** 2 + (listay[j+1] - listay[i]) ** 2)<ventana:
                index=index+1
            
            else:
                break
        esperas.append(index)
    
    maxima_espera= max(esperas)
#    print(maxima_espera)
    
    return maxima_espera
        

def calculo_Varianza_Difusion(listax, listay, divisiones):
    tam_list = len(listax)
    abcisas = range(1,10)
    abcisas = np.asarray(abcisas)
    num_por_div = math.floor(tam_list/divisiones)
    dividir = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
    
    mat_divx = dividir(listax,num_por_div)
    mat_divy = dividir(listay,num_por_div)
    msds = np.empty((divisiones,9))
    for i in range(divisiones):
        div_actx = mat_divx[i]
        div_acty = mat_divy[i]
        
        
        
        for j in range(1,9):
            final = num_por_div-j-1
            msd_act=0
            for k in range(final):
                msd_act += ((div_actx[k+j]-div_actx[k])**2)+((div_acty[k+j]-div_acty[k])**2)+np.finfo(float).eps
            msd_act = msd_act/final
#            if msd_act ==0:
#                msd_act =1*10**-10
            msds[i,j]=msd_act
    #procedemos ha calcular el coef de difusion de cada segmento de la ttrayectoria usando los msd calculados
    
    difusiones = np.empty((divisiones,1))
    
    
    for m in range(divisiones):

        p = np.polyfit(msds[m,:], abcisas, 1)
        difusiones[m,0]=p[0]
        
    varianza = np.var(difusiones) if np.var(difusiones) > 0  else np.finfo(float).eps
    varianza = np.log10(varianza)
    varianza = abs(varianza)
#    varianza = abs(varianza)
#    varianza = difusiones[0]/difusiones[1]
    return varianza
            
            
