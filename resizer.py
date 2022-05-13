from PIL import Image
import os

path = "datasetCarsDownsize/train"

for root, dirs, files in os.walk(path):
    for name in files:
        img = Image.open(root + "/" + name)

        resized = img.resize((128, 128), Image.LANCZOS)
        # BILINEAR, BICUBIC BRZO-KVALITETA--
        # LANCZOS SPORO-KVALITETA++

        resized.save(root + "/" + name, quality=100, subsampling=0)
        # quality=100, subsampling=0 NAJKVALITETNIJE

        print(root + "/" + name)
