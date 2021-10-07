import tensorflow as tf
from tensorflow.keras.regularizers import l2

umbral_max = 200
umbral_var = 40

#def my_network(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
#
#    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
##    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
#
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#    x = tf.keras.layers.Flatten()(x)
#
##    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)
#
#    return model

#def my_network(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
#
##    x = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
###    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
##
##
##    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#    x = tf.keras.layers.Flatten()(x)
#
##    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)
#
#    return model
#
#def my_network1(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
#
##    x = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
###    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
##
##
##    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
##
##    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#    x = tf.keras.layers.Flatten()(x)
#
##    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)
#
#    return model
#
#def my_network2(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
##
##    x = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
###    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
##
##
##    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
##    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#    x = tf.keras.layers.Flatten()(x)
#
##    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#    
#    x = tf.keras.layers.Dense(units=512, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.5)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)
#
#    return model
#
def my_network(input_shape, num_clases):
    input_tensor = tf.keras.Input(shape=input_shape)

#    x = tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
##    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
#
#
#    x = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
    
    


#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    

    x = tf.keras.layers.Flatten()(x)

#    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.5)(x)
    
    x = tf.keras.layers.Dense(units=512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)

    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.5)(x)



    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)




    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)




    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)

    return model



def my_network1(input_shape, num_clases):
    input_tensor = tf.keras.Input(shape=input_shape)
    max_espera_tensor = tf.keras.Input(shape=[1])
#    with tf.Session() as sess:
#        val_max = sess.run(max_espera_tensor)
#    print(val_max)
#    varianza_tensor = tf.keras.Input(shape=[1])

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)


    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)


    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

#    x = tf.keras.layers.Concatenate()([x,max_espera_tensor, varianza_tensor])
    
    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=4, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
#    if max_espera_tensor > umbral_max:
#        max_espera_tensor =1
#    else:
#        max_espera_tensor = max_espera_tensor/umbral_max
        
#    if varianza_tensor > umbral_var:
#        varianza_tensor = 1
#    else:
#        varianza_tensor = varianza_tensor/umbral_var
    
    

    x = tf.keras.layers.Concatenate()([x,max_espera_tensor])
    
#    x = tf.keras.layers.Dense(units=5, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)

    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)

#    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor, varianza_tensor], outputs=posteriors)
    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor], outputs=posteriors)
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)

    return model

def my_network2(input_shape, num_clases):
    input_tensor = tf.keras.Input(shape=input_shape)
    max_espera_tensor = tf.keras.Input(shape=[1])
#    varianza_tensor = tf.keras.Input(shape=[1])

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)


    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)


    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

#    x = tf.keras.layers.Concatenate()([x,max_espera_tensor, varianza_tensor])
    
    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    

    x = tf.keras.layers.Concatenate()([x,max_espera_tensor])
    
#    x = tf.keras.layers.Dense(units=5, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)

    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)

#    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor, varianza_tensor], outputs=posteriors)
    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor], outputs=posteriors)
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)

    return model

def my_network3(input_shape, num_clases):
    input_tensor = tf.keras.Input(shape=input_shape)
    max_espera_tensor = tf.keras.Input(shape=[1])
#    varianza_tensor = tf.keras.Input(shape=[1])

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)


    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)


    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

#    x = tf.keras.layers.Concatenate()([x,max_espera_tensor, varianza_tensor])
    
    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    

    x = tf.keras.layers.Concatenate()([x,max_espera_tensor])
    
#    x = tf.keras.layers.Dense(units=5, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)

    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)

#    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor, varianza_tensor], outputs=posteriors)
    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor], outputs=posteriors)
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)

    return model

def my_network4(input_shape, num_clases):
    input_tensor = tf.keras.Input(shape=input_shape)
    max_espera_tensor = tf.keras.Input(shape=[1])
#    varianza_tensor = tf.keras.Input(shape=[1])

    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)


    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)


    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)

#    x = tf.keras.layers.Concatenate()([x,max_espera_tensor, varianza_tensor])
    
    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    x = tf.keras.layers.Dense(units=10, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    

    x = tf.keras.layers.Concatenate()([x,max_espera_tensor])
    
#    x = tf.keras.layers.Dense(units=5, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)
    
    

    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)

#    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor, varianza_tensor], outputs=posteriors)
    model = tf.keras.Model(inputs=[input_tensor, max_espera_tensor], outputs=posteriors)
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)

    return model

#def my_network2(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
#
#    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
##    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
#
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=1536, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#    x = tf.keras.layers.Flatten()(x)
#
#    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)
#
#    return model
#
#def my_network3(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
#
#    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
##    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
#
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
##    x = tf.keras.layers.Conv2D(filters=1536, kernel_size=(3, 3), padding='same', activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#    x = tf.keras.layers.Flatten()(x)
#
#    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.3)(x)
#    
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)
#
#    return model
#
#def my_network4(input_shape, num_clases):
#    input_tensor = tf.keras.Input(shape=input_shape)
#
#    x = tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
##    kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
#
#
#    x = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#
#    x = tf.keras.layers.Conv2D(filters=768, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#    x = tf.keras.layers.Conv2D(filters=1536, kernel_size=(3, 3), padding='same', activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#    
#
#
#
#    x = tf.keras.layers.Flatten()(x)
#
##    x = tf.keras.layers.Dense(units=1024, activation='relu')(x)
##    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
##    x = tf.keras.layers.Dropout(rate=0.3)(x)
#    
#
#    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)
#
#
#    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)
#
#
#
#    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
#    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
#    x = tf.keras.layers.Dropout(rate=0.3)(x)
#
#
#
#    posteriors = tf.keras.layers.Dense(units=num_clases, activation='softmax')(x)
#
#    model = tf.keras.Model(inputs=input_tensor, outputs=posteriors)

    return model