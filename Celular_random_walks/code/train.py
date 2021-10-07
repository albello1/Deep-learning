import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import dataset
import networks
import matplotlib.pyplot as plt
import pandas as pd
#import cv2
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#from pruebas import _parse_function 
import glob
from dataset import parse_tfrecord


datasets_directory = '/home/jmgarcia/Albert/RW/dataset'
#tf.enable_eager_execution()

#datasets_directory= 'C:\ALBERT\Programas Python\gorka\challenge\codigo_javi_nuevo\dataset'
image_dimension = 200

# Run this only once if you need to create the datasets
#dataset.create_datasets(output_directory=os.path.join(datasets_directory, 'train'), image_dimension=image_dimension,
#                         is_training=True)
#dataset.create_datasets(output_directory=os.path.join(datasets_directory, 'validation'), image_dimension=image_dimension,
#                         is_training=False)


# Get the Tensorflow Datasets




train_dataset = dataset.train_dataset(tfrecords_directory=os.path.join(datasets_directory, 'train'), batch_size=5)
validation_dataset = dataset.validation_dataset(tfrecords_directory=os.path.join(datasets_directory, 'validation'),
                                                batch_size=5)

#for (image,max_espera,varianza), label in train_dataset:
#    print(image.shape)
#    print(max_espera)
#    print(varianza)
#    break

#for images, labels in train_dataset:
#    image = images[0,...]
#    plt.figure()
#    plt.imshow(image[:,:,0])
#    plt.show()
#    
#    plt.figure()
#    plt.imshow(image[:,:,1])
#    plt.show()
#    
##    dif = cv2.subtract(image[:,:,0],image[:,:,1])
##    if not np.any(dif):
##        print("las imagenes son iguales")
##    else:
##        print("las imagenes son distintas")
#            
#    im_rest= image[:,:,0]-image[:,:,1]
#    plt.figure()
#    plt.imshow(im_rest)
#    plt.show()

my_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
my_loss = tf.keras.losses.SparseCategoricalCrossentropy()
my_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

#model = networks.my_network(input_shape=(image_dimension, image_dimension, 1), num_clases=5)
#model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
#model.summary()
#
#
#
##history=model.fit(x=train_dataset, epochs=2, steps_per_epoch=5, validation_data=validation_dataset,
##          validation_steps=5)
#history=model.fit(x=train_dataset, epochs=250, steps_per_epoch=500, validation_data=validation_dataset,
#          validation_steps=500)
#pd.DataFrame.from_dict(history.history).to_csv('history.csv',index=False)
##
##
###y_pred = np.argmax(model.predict(validation_dataset), axis=-1)
##
##mat = confusion_matrix(validation_dataset,y_pred)
##plot_confusion_matrix(conf_mat=mat)
##
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['sparse_categorical_accuracy'])
#plt.plot(history.history['val_sparse_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("accuracy.png")
#plt.close()
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("loss.png")
#plt.close()


#features,label = iter(validation_dataset)
#
#iterator = tf.compat.v1.data.make_one_shot_iterator(validation_dataset)
#features, label = iterator.get_next()
#
#for element in validation_dataset:
#    print(element)
#
#tam_label = np.shape(label)
#print(tam_label)
#print(label)
#
#
#
#y_pred=model.predict(features)
#y_pred_new = np.argmax(y_pred,1)
##tam_pred = np.shape(y_pred_new)
#print(y_pred)
#
#mat = confusion_matrix(label,y_pred_new)
#plot_confusion_matrix(conf_mat=mat)
#plt.savefig("conf_mat.png")
#plt.close()


##########################################3

model = networks.my_network1(input_shape=(image_dimension, image_dimension, 2), num_clases=5)
model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
model.summary()


##############################################################3






#history=model.fit(x=train_dataset, epochs=50, steps_per_epoch=500, validation_data=validation_dataset,
#          validation_steps=500)
history=model.fit(x=train_dataset, epochs=2, steps_per_epoch=5, validation_data=validation_dataset,
          validation_steps=5)
model.save('C:\ALBERT\Programas Python\gorka\challenge\codigo_javi_nuevo\modelo.h5')
#pd.DataFrame.from_dict(history.history).to_csv('history1.csv',index=False)
#
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['sparse_categorical_accuracy'])
#plt.plot(history.history['val_sparse_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("accuracy1.png")
#plt.close()
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("loss1.png")
#plt.close()




#AQUI VA LO BUENO DE LA MATRIZ DE CONFUSION

#tfrecord_path = glob.glob(os.path.join(os.path.join(datasets_directory, 'validation'), '*.test_tfrecord'))
#ds = tf.data.TFRecordDataset(tfrecord_path)
#ds = ds.map(map_func=parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#ds = dataset.validation_dataset(tfrecords_directory=os.path.join(datasets_directory, 'validation'),
#                                                repeat=False, batch_size=5)
#
#ground_truth =[]
#for element in ds:
#    ground_truth.append(element[1])
#
#y_pred=model.predict(ds)
#y_pred_new = np.argmax(y_pred,1)
#
#
#mat = confusion_matrix(ground_truth,y_pred_new)
#plot_confusion_matrix(conf_mat=mat)
#plt.savefig("conf_mat.png")
#print(mat)
#plt.close()

#AQUI ACABA LO DE LA MATRIZ DE CONFUSION





#iterator = tf.compat.v1.data.make_one_shot_iterator(validation_dataset)
#features, label = iterator.get_next()
#

#for element in validation_dataset:
#    lista.append(element[0])
#
#tam_label = np.shape(label)
#print(tam_label)
#print(label)
#
#
#
#y_pred=model.predict(features)
#y_pred_new = np.argmax(y_pred,1)
##tam_pred = np.shape(y_pred_new)
#print(y_pred)
#
#mat = confusion_matrix(label,y_pred_new)
#plot_confusion_matrix(conf_mat=mat)
#plt.savefig("conf_mat.png")
#plt.close()


# Map features and labels with the parse function.
#iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
#features, label = iterator.get_next()
#
#print(features)
#ds = ds.map(_parse_function)
#n = ds.make_one_shot_iterator().get_next()
#
#sess = tf.compat.v1.Session()
#
#tot = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(os.path.join(datasets_directory, 'validation')))
#
#output_features=[]
#output_labels=[]
#
#for i in range(0,tot-1):
#  value=sess.run(n)
#  output_features.append(value[0]['terms'])
#  output_labels.append(value[1])

#model.save("my_model")

#model = networks.my_network2(input_shape=(image_dimension, image_dimension, 2), num_clases=5)
#model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
#model.summary()
#history=model.fit(x=train_dataset, epochs=250, steps_per_epoch=500, validation_data=validation_dataset,
#          validation_steps=500)
#pd.DataFrame.from_dict(history.history).to_csv('history2.csv',index=False)
#
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['sparse_categorical_accuracy'])
#plt.plot(history.history['val_sparse_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("accuracy2.png")
#plt.close()
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("loss2.png")
#plt.close()
#
#model = networks.my_network3(input_shape=(image_dimension, image_dimension, 2), num_clases=5)
#model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
#model.summary()
#history=model.fit(x=train_dataset, epochs=250, steps_per_epoch=500, validation_data=validation_dataset,
#          validation_steps=500)
#pd.DataFrame.from_dict(history.history).to_csv('history3.csv',index=False)
#
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['sparse_categorical_accuracy'])
#plt.plot(history.history['val_sparse_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("accuracy3.png")
#plt.close()
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("loss3.png")
#plt.close()
#
#model = networks.my_network4(input_shape=(image_dimension, image_dimension, 2), num_clases=5)
#model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
#model.summary()
#history=model.fit(x=train_dataset, epochs=250, steps_per_epoch=500, validation_data=validation_dataset,
#          validation_steps=500)
#pd.DataFrame.from_dict(history.history).to_csv('history4.csv',index=False)
#
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['sparse_categorical_accuracy'])
#plt.plot(history.history['val_sparse_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("accuracy4.png")
#plt.close()
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("loss4.png")
#plt.close()

#model = networks.my_network3(input_shape=(image_dimension, image_dimension, 1), num_clases=5)
#model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
#model.summary()
#history=model.fit(x=train_dataset, epochs=200, steps_per_epoch=500, validation_data=validation_dataset,
#          validation_steps=500)
#
#print(history.history.keys())
## summarize history for accuracy
#plt.plot(history.history['sparse_categorical_accuracy'])
#plt.plot(history.history['val_sparse_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("accuracy1.png")
#plt.close()
##plt.show()
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig("loss1.png")
#plt.close()
##
##model = networks.my_network2(input_shape=(image_dimension, image_dimension, 1), num_clases=5)
##model.compile(optimizer=my_optimizer, loss=my_loss, metrics=[my_accuracy])
##model.summary()
##history=model.fit(x=train_dataset, epochs=200, steps_per_epoch=500, validation_data=validation_dataset,
##          validation_steps=500)
##
##print(history.history.keys())
### summarize history for accuracy
##plt.plot(history.history['sparse_categorical_accuracy'])
##plt.plot(history.history['val_sparse_categorical_accuracy'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.savefig("accuracy2.png")
##plt.close()
###plt.show()
### summarize history for loss
##plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.savefig("loss2.png")
##plt.close()
#
## Get the model
## model = networks.my_network(input_shape=(image_dimension, image_dimension, 2), num_clases=5)
