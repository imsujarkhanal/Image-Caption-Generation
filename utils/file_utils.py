import os

def validate_image_path(image_path, base_dir):
    full_path = os.path.join(base_dir, image_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")
    return full_path