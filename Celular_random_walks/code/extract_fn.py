# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 01:20:03 2020

@author: Albert
"""
import tensorflow as tf

def extract_fn(data_record):
    features = {
        # Extract features using the keys set during creation
        "image/class/label":    tf.FixedLenFeature([], tf.int64),
        "image/encoded":        tf.VarLenFeature(tf.string),
    }
    sample = tf.parse_single_example(data_record, features)
    image = tf.decode_raw(sample['image/encoded'], tf.float32)
    label = sample['image/class/label']

    return image, label