import base64
import io
from PIL import Image

# Function to encode a single image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

#def resize_encode_image(image_path, resize=None):
    # Open image
#    img = Image.open(image_path)
    
    # Resize if requested
#    if resize is not None:
#        img.thumbnail((resize, resize))
        
    # Encode bytes to base64 string
 #   return base64.b64encode(img.tobytes()).decode("utf-8")

def resize_encode_image(image_path, resize=None):
    # Open image
    img = Image.open(image_path)
    
    # Resize if requested
    if resize is not None:
        img.thumbnail((resize, resize))
    
    # Save to bytes buffer in standard format (PNG)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(img_bytes).decode("utf-8")