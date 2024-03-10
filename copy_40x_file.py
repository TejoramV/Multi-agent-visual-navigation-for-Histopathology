import os
import shutil

# Source and destination folders
source_folder = "C:/Users/stlp/Desktop/Linda/convert2tif/"
destination_folder = "C:/Users/stlp/Desktop/Linda/40X"

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file ends with "_x40_z0.tif"
    if filename.endswith("_x40_z0.tif"):
        # If the file ends with the desired string, copy it to the destination folder
        source_file_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        shutil.copyfile(source_file_path, destination_file_path)

