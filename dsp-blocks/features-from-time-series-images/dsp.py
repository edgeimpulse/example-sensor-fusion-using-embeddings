import argparse
import json
import math
import numpy as np
import sys
import io, base64
from PIL import Image

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq):
    if (implementation_version != 1):
        raise Exception('implementation_version should be 1')

    graphs = []
    all_features = []
    # print(len(axes))
    # print(len(raw_data))
    # print(sampling_freq)

    # Hardcode the size of the image
    w = 100
    h = 100
    
    raw_data = raw_data.astype(dtype=np.uint8).view(dtype=np.uint8)
    
    # features is a 1D array, reshape so we have a matrix
    image_array = raw_data.reshape(100, 100, 3)

    # Normalize and make sure we never divide by 0
    if(image_array.max() != 0):
        image_array = image_array / image_array.max()
    else:
        image_array = np.zeros((w, h, 3))
    # print(image_array)

    # Generate features, flatten and set to float32 values 
    features = np.float32(image_array.reshape(h*w*3))

    # Create image
    im = Image.fromarray(np.uint8(image_array*255), 'RGB')
            
    # Increase image size for the rendering in Edge Impulse studio        
    im = im.resize((256,256), Image.NEAREST)

    # Save the image to a buffer, and base64 encode the buffer
    with io.BytesIO() as buf:
            im.save(buf, format='png')
            buf.seek(0)
            image = (base64.b64encode(buf.getvalue()).decode('ascii'))

            # append as a new graph
            graphs.insert(0,{
                'name': 'Computed matrix',
                'image': image,
                'imageMimeType': 'image/png',
                'type': 'image'
            })

    return {
        'features': features,
        'graphs': graphs,
        # if you use FFTs then set the used FFTs here (this helps with memory optimization on MCUs)
        'fft_used': [],
        'output_config': {
            # type can be 'flat', 'image' or 'spectrogram'
            'type': 'flat',
            'shape': {
                # shape should be { width, height, channels } for image, { width, height } for spectrogram
                'width': len(features)
            }
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Returns raw data')
    parser.add_argument('--features', type=str, required=True,
                        help='Axis data as a flattened WAV file (pass as comma separated values)')
    parser.add_argument('--axes', type=str, required=True,
                        help='Names of the axis (pass as comma separated values)')
    parser.add_argument('--frequency', type=float, required=True,
                        help='Frequency in hz')
    parser.add_argument('--draw-graphs', type=bool, required=True,
                        help='Whether to draw graphs')

    args = parser.parse_args()

    raw_features = np.array([float(item.strip()) for item in args.features.split(',')])
    raw_axes = args.axes.split(',')

    try:
        processed = generate_features(1, False, raw_features, args.axes, args.frequency)

        print('Begin output')
        # print(json.dumps(processed))
        print('End output')
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)