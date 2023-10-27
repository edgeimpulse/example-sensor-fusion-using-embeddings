import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization,
    Conv2D, GlobalMaxPooling2D, Lambda, GlobalAveragePooling2D)
from tensorflow.keras.models import Sequential, Model
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert a TensorFlow saved model to embeddings.')
parser.add_argument('--input', required=True, help='Path to the input folder containing the saved model and input.npy.')
parser.add_argument('--output', required=True, help='Path to the folder where the converted models will be saved.')
args = parser.parse_args()

curr_dir = os.path.dirname(os.path.realpath(__file__))

base_model = tf.keras.models.load_model(os.path.join(args.input, 'saved_model'))

input_shape = tuple(base_model.layers[0].get_input_at(0).get_shape().as_list()[1:])
output_shape = tuple(base_model.layers[-2].get_output_at(0).get_shape().as_list())
print('Input shape', input_shape)
print('Output shape (embeddings)', output_shape)

model = Sequential()
model.add(InputLayer(input_shape=input_shape, name='x_input'))
model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output))
model.add(Flatten())

def convert_float32(concrete_func, keras_model, dir_path, filename):
    try:
        print('Converting TensorFlow Lite float32 model...', flush=True)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        # Restrict the supported types to avoid ops that are not TFLM compatible
        converter.target_spec.supported_types = [
            tf.dtypes.float32,
            tf.dtypes.int8
        ]
        tflite_model = converter.convert()
        open(os.path.join(dir_path, filename), 'wb').write(tflite_model)
        return tflite_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite float32 model:')
        print(err)

def convert_int8_io_int8(concrete_func, keras_model, dataset_generator,
                         dir_path, filename, disable_per_channel = False):
    try:
        print('Converting TensorFlow Lite int8 quantized model...', flush=True)
        converter_quantize = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
        if disable_per_channel:
            converter_quantize._experimental_disable_per_channel = disable_per_channel
            print('Note: Per channel quantization has been automatically disabled for this model. '
                  'You can configure this in Keras (expert) mode.')
        converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quantize.representative_dataset = dataset_generator
        # Force the input and output to be int8
        converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Restrict the supported types to avoid ops that are not TFLM compatible
        converter_quantize.target_spec.supported_types = [tf.dtypes.int8]
        converter_quantize.inference_input_type = tf.int8
        converter_quantize.inference_output_type = tf.int8
        tflite_quant_model = converter_quantize.convert()
        open(os.path.join(dir_path, filename), 'wb').write(tflite_quant_model)
        return tflite_quant_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite int8 quantized model:')
        print(err)

def get_concrete_function(keras_model, input_shape):
    # To produce an optimized model, the converter needs to see a static batch dimension.
    # At this point our model has an unspecified batch dimension, so we need to set it to 1.
    # See: https://github.com/tensorflow/tensorflow/issues/42286#issuecomment-681183961
    input_shape_with_batch = (1,) + input_shape
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(input_shape_with_batch, keras_model.inputs[0].dtype))
    return concrete_func

# Declare a generator that can feed the TensorFlow Lite converter during quantization
def representative_dataset_generator(validation_dataset):
    def gen():
        for i in range(0, validation_dataset.shape[0]):
            yield [tf.convert_to_tensor([validation_dataset[i]])]
    return gen

def convert_to_tf_lite(model, dir_path,
                      validation_dataset, model_input_shape, model_filenames_float,
                      model_filenames_quantised_int8, disable_per_channel = False):
    dataset_generator = representative_dataset_generator(validation_dataset)
    concrete_func = get_concrete_function(model, model_input_shape)

    tflite_model = convert_float32(concrete_func, model, dir_path, model_filenames_float)
    tflite_quant_model = convert_int8_io_int8(concrete_func, model, dataset_generator,
                                              dir_path, model_filenames_quantised_int8,
                                              disable_per_channel)

    return model, tflite_model, tflite_quant_model

convert_to_tf_lite(model, args.output, np.load(os.path.join(args.input, 'input.npy')),
    input_shape, 'embeddings.tflite', 'embeddings-quantized.tflite')