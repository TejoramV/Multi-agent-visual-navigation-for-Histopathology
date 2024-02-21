OPENSLIDE_PATH = r'C:/Users/stlp/Downloads/openslide-win64-20231011/openslide-win64-20231011/bin'
import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from openslide import OpenSlide
from PIL import Image
Image.MAX_IMAGE_PIXELS = 689733632

from openslide.deepzoom import DeepZoomGenerator


img_path = "C:/Users/stlp/Desktop/Linda/convert2tif/MP_0001_x0.625_z0.tif"
slide = openslide.open_slide(img_path)

print("Level count:", slide.level_count)
print("Dimensions at level 0:", slide.dimensions)
print("Number of levels:", slide.level_count)
print("Level dimensions:", [slide.level_dimensions[i] for i in range(slide.level_count)])
print(slide)
