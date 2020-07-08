import coremltools

# Load a model, lower its precision, and then save the smaller model.
model_spec = coremltools.utils.load_spec('Smile_Detection_Model/SPOOF__MOBILENET_21_4_2020.mlmodel')
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, 'SPOOF__MOBILENET_21_4_2020_reduced.mlmodel')