from keras.models import load_model
import coremltools

# model.save('trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')


your_model = coremltools.converters.keras.convert('D:/Python Workspace/COREML_TFLITE/SPOOF__MOBILENET_21_4_2020.h5',input_names=['image'],image_input_names='image',output_names=['output'],image_scale=1 / 255.0,predicted_feature_name='close_open_eye')
your_model.short_description = 'Predicts the open/close eyes present in an image of a human face.'
your_model.input_description['image'] = '150X150 Image of Human Face!'
your_model.output_description['output'] = 'Predicted open/closed eyes 0.5-1-> Real Face  0-0.5->Fake Face'

your_model.author = 'Inficare AI Team'
your_model.license = 'Inficare'

your_model.save('D:/Python Workspace/COREML_TFLITE/Smile_Detection_Model/SPOOF__MOBILENET_21_4_2020.mlmodel')