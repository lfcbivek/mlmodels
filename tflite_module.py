import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model_file('EYE_BLINK__MOBILENET_23_4_2020.h5') 
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tfmodel = converter.convert() 
open ("D:/Python Workspace/EYE_BLINK__MOBILENET_23_4_2020.tflite" , "wb") .write(tfmodel)