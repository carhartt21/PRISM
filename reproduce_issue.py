
import os
from PIL import Image
from utils.image_io import load_image
import torch

def create_truncated_image(filename):
    # Create a valid image
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(filename)
    
    # Truncate it
    with open(filename, 'rb') as f:
        data = f.read()
    
    # Write back only half the data
    with open(filename, 'wb') as f:
        f.write(data[:len(data)//2])

def test_load_truncated():
    filename = "truncated_test.png"
    create_truncated_image(filename)
    
    print(f"Created truncated image: {filename}")
    
    try:
        tensor = load_image(filename)
        print("Successfully loaded truncated image!")
        print(f"Tensor shape: {tensor.shape}")
    except Exception as e:
        print(f"Caught expected exception: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_load_truncated()
