from PIL import Image
import os

dataset_path = "dataset"  # Change if needed


def remove_icc_profiles(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # Convert to standard RGB
                    img.save(img_path, "PNG")  # Overwrite without ICC


# Apply the fix
remove_icc_profiles(dataset_path)
print("ICC profiles removed from all PNG images.")
