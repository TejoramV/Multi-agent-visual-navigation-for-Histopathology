import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

excel_path = 'C:/Users/stlp/Desktop/Linda/pathologists_viewport_40x.csv' 
df = pd.read_csv(excel_path)
zoomLevel = df["zoomLevel"]
zoomLevel = list(zoomLevel)
unique_zoomLevel = sorted(list(set(zoomLevel)))
patch_number_to_zoom_lvl=[]
for i in unique_zoomLevel:
    patch_number_to_zoom_lvl.append(zoomLevel.count(i))

pathologist_id = df["pathologist_id"]
pathologist_id = list(pathologist_id)
unique_pathologist_id = sorted(list(set(pathologist_id)))
patch_number_per_wsi=[]
for i in unique_pathologist_id:
    patch_number_per_wsi.append(pathologist_id.count(i))

patch_number_min = min(patch_number_per_wsi)
patch_number_max =max(patch_number_per_wsi)
patch_number_mean = np.average(patch_number_per_wsi)
patch_number_std = np.std(patch_number_per_wsi)
# print(patch_number_min,patch_number_max,patch_number_mean,patch_number_std) #
# print(patch_number_to_zoom_lvl) # y-axis patch number , x-axis zoom 1- 60

patch_number_to_zoom_lvl_probabilities = np.array(patch_number_to_zoom_lvl) / np.sum(patch_number_to_zoom_lvl)
# # result = ', '.join(map(str, patch_number_to_zoom_lvl_probabilities))
# # print(result)
print(patch_number_to_zoom_lvl_probabilities)


###########################
## TO calculate width and height
## 

width_dist =[]
height_dist = []

width_min=[]
width_max=[]
height_min=[]
height_max=[]
width_avg = []
height_avg = []
width_std = []
height_std = []
for i in range(1,61):
    subset_data = df[df['zoomLevel']==i]
    width_dist.append((min(subset_data['width']), max(subset_data['width']), np.average(subset_data['width']), np.std(subset_data['width']) ))
    height_dist.append((min(subset_data['height']), max(subset_data['height']), np.average(subset_data['height']), np.std(subset_data['height']) ))
    width_min.append(min(subset_data['width']))
    width_max.append(max(subset_data['width']))  
    height_min.append(min(subset_data['height']))
    height_max.append(max(subset_data['height']))
    width_avg.append(np.average(subset_data['width']))
    height_avg.append(np.average(subset_data['height']))
    width_std.append(np.std(subset_data['width']))
    height_std.append(np.std(subset_data['height']))

data = patch_number_to_zoom_lvl
x_values = range(1,61)
plt.plot(x_values, data)
plt.xlabel('zoom-level')
plt.ylabel('Patch_frequency')
plt.title('Plot')

plt.show()
