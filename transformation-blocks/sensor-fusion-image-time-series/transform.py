from PIL import Image
import json
import os
import time
import random 

# Directory containing your images
image_directory = "input"  # Replace with your image directory path
output_directory = "output"  # Replace with your desired output directory path

# Ensure the output directory exists, create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Empty signature (all zeros). HS256 gives a 32-byte signature, and we encode in hex, so we need 64 characters here.
emptySignature = ''.join(['0'] * 64)

# Get a list of image files in the image_directory
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif"))]

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    output_json_path = os.path.join(output_directory, os.path.splitext(image_file)[0] + ".json")

    # Load the image
    image = Image.open(image_path)

    # Resize the image to 100x100 pixels
    image = image.resize((100, 100), Image.ANTIALIAS)

    # Get the pixel data as a list of tuples (R, G, B)
    pixel_data = list(image.getdata())
    # Generate random values between -20 and 20 for the additional sensor
    additional_sensor_values = [random.uniform(-10, 20) for _ in pixel_data]

    # Prepare the JSON structure
    json_data = {
        "protected": {
            "ver": "v1",
            "alg": "HS256",
            "iat": time.time()  
        },
        "signature": emptySignature,
        "payload": {
            "device_name": "transformation_block",
            "device_type": "IMAGE_SENSOR",
            "interval_ms": 10,
            "sensors": [
                {"name": "Red", "units": "pixel_value"},
                {"name": "Green", "units": "pixel_value"},
                {"name": "Blue", "units": "pixel_value"},
                {"name": "Sensor", "units": "unit"}
            ],
            "values": [list(pixel) + [v] for pixel, v in zip(pixel_data, additional_sensor_values)]
        }
    }

    # Convert the JSON structure to a JSON string
    json_string = json.dumps(json_data, indent=2)

    # Save the JSON to a file
    with open(output_json_path, "w") as json_file:
        json_file.write(json_string)