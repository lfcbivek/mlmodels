import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model_file('EYE_BLINK__MOBILENET_23_4_2020.h5') 
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tfmodel = converter.convert() 
open ("D:/Python Workspace/EYE_BLINK__MOBILENET_23_4_2020.tflite" , "wb") .write(tfmodel)

import tensorflow as tf




import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="D:/Python Workspace/EYE_BLINK__MOBILENET_23_4_2020.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details()[0]['shape'])  
print(interpreter.get_input_details()[0]['dtype']) 

print(interpreter.get_output_details()[0]['shape'])  
print(interpreter.get_output_details()[0]['dtype'])