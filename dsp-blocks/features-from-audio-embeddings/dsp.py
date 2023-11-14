import argparse
import json
import numpy as np
import os, sys
import math
import tensorflow as tf

import pathlib
ROOT = pathlib.Path(__file__).parent
sys.path.append(str(ROOT / '..'))
from common.errors import ConfigurationError
from common import graphing

curr_dir = os.path.dirname(os.path.realpath(__file__))

# Load our SpeechPy fork
MODULE_PATH = os.path.join(curr_dir, 'third_party', 'speechpy', '__init__.py')
MODULE_NAME = 'speechpy'
import importlib
import sys
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
speechpy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speechpy
spec.loader.exec_module(speechpy)

interpreter = tf.lite.Interpreter(model_path=os.path.join(curr_dir, 'embeddings-quantized.tflite'))
interpreter.allocate_tensors()

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.
    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data
    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.around(data)
        data = data.astype(np.int8)
    return tf.convert_to_tensor(data)

def process_output(output_details, data):
    """Prepares an output (dequantizes)
    Args:
        output_details: The result of calling interpreter.get_output_details()
        data (numpy array): The raw output data
    Returns:
        A tensor object representing the output, quantized if necessary
    """
    if output_details[0]['dtype'] is np.int8:
        data = data.astype(np.float32)
        scale = output_details[0]['quantization'][0]
        zero_point = output_details[0]['quantization'][1]
        data = (data - zero_point) * scale
    return data

def invoke(interpreter, item, specific_input_shape):
    """Invokes the Python TF Lite interpreter with a given input
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)
    if specific_input_shape:
        item_as_tensor = tf.reshape(item_as_tensor, specific_input_shape)
    # Add batch dimension
    item_as_tensor = tf.expand_dims(item_as_tensor, 0)
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details


def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq):
    frame_length = 0.032
    frame_stride = 0.024
    fft_length = 128
    noise_floor_db = -85

    if (implementation_version != 1 and implementation_version != 2 and implementation_version != 3):
        raise ConfigurationError('implementation_version should be 1, 2 or 3')

    if (not math.log2(fft_length).is_integer()):
        raise ConfigurationError('FFT length must be a power of 2')

    if (len(axes) != 1):
        raise ConfigurationError('Spectrogram blocks only support a single axis, ' +
            'create one spectrogram block per axis under **Create impulse**')

    fs = sampling_freq

    # reshape first
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    graphs = []

    width = 0
    height = 0

    for ax in range(0, len(axes)):
        signal = raw_data[:,ax]

        if implementation_version >= 3:
            # Rescale to [-1, 1] and add preemphasis
            if np.any((signal < -1) | (signal > 1)):
                signal = (signal / 2**15).astype(np.float32)

        sampling_frequency = fs

        s = np.array(signal).astype(float)

        numframes, _, __ = speechpy.processing.calculate_number_of_frames(
            s,
            implementation_version=implementation_version,
            sampling_frequency=sampling_frequency,
            frame_length=frame_length,
            frame_stride=frame_stride,
            zero_padding=False)

        if (numframes < 1):
            raise ConfigurationError('Frame length is larger than your window size')

        if (numframes > 500):
            raise ConfigurationError('Number of frames is larger than 500 (' + str(numframes) + '), ' +
                'increase your frame stride')

        frames = speechpy.processing.stack_frames(
            s,
            implementation_version=implementation_version,
            sampling_frequency=sampling_frequency,
            frame_length=frame_length,
            frame_stride=frame_stride,
            filter=lambda x: np.ones((x,)),
            zero_padding=False)

        power_spectrum = speechpy.processing.power_spectrum(frames, fft_length)

        if implementation_version < 3:
            power_spectrum = (power_spectrum - np.min(power_spectrum)) / (np.max(power_spectrum) - np.min(power_spectrum))
            power_spectrum[np.isnan(power_spectrum)] = 0
        else:
            # Clip to avoid zero values
            power_spectrum = np.clip(power_spectrum, 1e-30, None)
            # Convert to dB scale
            # log_mel_spec = 10 * log10(mel_spectrograms)
            power_spectrum = 10 * np.log10(power_spectrum)

            power_spectrum = (power_spectrum - noise_floor_db) / ((-1 * noise_floor_db) + 12)
            power_spectrum = np.clip(power_spectrum, 0, 1)

        flattened = power_spectrum.flatten()
        features = np.concatenate((features, flattened))

        width = np.shape(power_spectrum)[0]
        height = np.shape(power_spectrum)[1]

        if draw_graphs:
            # make visualization too
            power_spectrum = np.swapaxes(power_spectrum, 0, 1)
            image = graphing.create_sgram_graph(sampling_freq, frame_length, frame_stride, width, height, power_spectrum)

            graphs.append({
                'name': 'Spectrogram',
                'image': image,
                'imageMimeType': 'image/svg+xml',
                'type': 'image'
            })

    # now that we have features we're gonna pass it through the NN to get embeddings
    output, output_details = invoke(interpreter, features, None)
    output = process_output(output_details, output[0]).tolist()

    return {
        'features': output,
        'graphs': graphs,
        'fft_used': [ fft_length ],
        'output_config': {
            'type': 'flat',
            'shape': {
                'width': len(output)
            }
        }
    }

def get_tflite_implementation(implementation_version, input_shape, axes, sampling_freq):
    with open(os.path.join(curr_dir, "embeddings-quantized.tflite"), 'rb') as f:
        return f.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MFCC script for audio data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened WAV file (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--draw-graphs', type=lambda x: (str(x).lower() in ['true','1', 'yes']), required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(2, args.draw_graphs, raw_features, raw_axes, args.frequency,
            args.frame_length, args.frame_stride, args.num_filters, args.fft_length,
             args.low_frequency, args.high_frequency, args.win_size, args.noise_floor_db)

        print('Begin output')
        print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
