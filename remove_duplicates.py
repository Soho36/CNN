import os
import hashlib


def find_duplicates(directory):
    hashes = {}
    duplicates = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                duplicates.append(filepath)  # Found a duplicate
            else:
                hashes[file_hash] = filepath

    return duplicates


# Example usage:
duplicates = find_duplicates("E:\\YandexDisk\\Desktop_Zal\\CNN sets\\hold")
print("Duplicates:", duplicates)

# Optionally, delete duplicates
# for dup in duplicates:
#     os.remove(dup)
