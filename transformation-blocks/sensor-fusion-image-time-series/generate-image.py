from PIL import Image

# Define the image size
width, height = 100, 100

# Create a new RGB image with a white background
image = Image.new("RGB", (width, height), "white")

# Save the image to a file
image.save("output.png")

print("Image created and saved as 'output.png'")