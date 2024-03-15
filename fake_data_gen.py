import pandas as pd
import numpy as np
from glob import glob
import cv2
import random
import csv
import tifffile as tifi #pip install imagecodecs
import os

#write explaination
#fix delete bg
#run for all images
def refine_segmentation(image, mask, kernel_size=15):
    # Create a morphological kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)    
    # Close small holes in the foreground mask: Dilation followed by Erosion
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closing

def segment_background_foreground(image, output_path=None):
    # Load the image
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Define the range for background (white areas)
    # These thresholds might need adjustment for different image types
    lower = np.array([200, 200, 200], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")

    # Create a mask that identifies the background
    background_mask = cv2.inRange(image_rgb, lower, upper)
    # Invert mask to get the foreground
    foreground_mask = cv2.bitwise_not(background_mask)

    refined_foreground_mask = refine_segmentation(image_rgb, foreground_mask)

    # Apply the foreground mask to the image
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=refined_foreground_mask)

    # If an output path is provided, save the segmented image
    if output_path is not None:
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    return segmented_image

def zoom_in(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height // scale_factor)
    new_width = int(width // scale_factor)
    zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return zoomed_image

def crop_image(image, start_x, start_y, width, height):
    cropped_image = image[start_y:start_y+height, start_x:start_x+width]
    return cropped_image

def random_WH_generator(zoom):
    #in order(min,max,mean, std)
    width_per_zoom_dist = [(30160, 75960, 50200.009090909094, 8350.353129708135), (13333, 51200, 23994.131710973994, 5651.159700761149), (10053, 25600, 17491.4258436214, 4725.238913259465), (6260, 30267, 12373.3058749679, 2974.321933645031), (5008, 15360, 10109.232345488597, 2832.348774297481), (4173, 12800, 8924.84534588917, 2351.1823465576263), (3577, 10971, 7590.048582081121, 2103.355889493203), (3130, 9600, 6679.439247943596, 1730.741932702139), (3351, 8533, 6166.890958478852, 1579.9896095713357), (2501, 10240, 4876.923762272458, 1206.3226283029576), (2742, 6982, 4814.449531163817, 1369.0247577899622), (2256, 6400, 4628.1753027316245, 1248.022006898369), (2320, 9959, 4233.648342495419, 1084.526343500683), (2154, 5486, 3882.482453925776, 1050.8946959433017), (2011, 5120, 3626.128509472723, 966.6702588415194), (1885, 4800, 3494.5055555555555, 870.5243137827973), (1774, 4518, 3295.2771340742493, 797.5030376775736), (1676, 4267, 3172.8042003231017, 782.1140142529856), (1587, 4042, 2997.730824066153, 734.521710831931), (1508, 5120, 2473.215886283677, 626.3472717641893), (1436, 3657, 2566.574537037037, 733.8785365993933), (1371, 3491, 2509.308101939901, 710.4769345244252), (1311, 3339, 2263.0715846994535, 605.4155857653919), (1257, 3200, 2288.7783783783784, 578.2323238819552), (1206, 3072, 2261.7378223495703, 545.2216210648961), (1223, 2954, 2066.3660079051383, 560.2773519405384), (1117, 2844, 2189.6773333333335, 556.3733565567479), (1077, 2743, 1965.331189710611, 505.7537477230234), (1040, 2648, 1789.9757994814174, 491.6762777238617), (1311, 2560, 2004.4685990338164, 526.9997733313307), (1251, 2477, 1824.4465240641712, 488.2634360355087), (994, 2400, 1739.438063063063, 423.2411972252709), (964, 2327, 1815.7845303867402, 464.56035928804005), (1156, 2259, 1792.5721003134797, 458.7425469970812), (1108, 2194, 1657.285046728972, 457.85100223073596), (1077, 2133, 1600.359, 260.36292385629713), (1063, 2076, 1786.2440677966101, 360.93769792977133), (1021, 2021, 1371.6290801186944, 322.25298149150257), (1008, 1969, 1439.8760330578511, 400.4598404406501), (907, 2560, 1312.550142074988, 305.52492273070993), (974, 1873, 1495.4689119170985, 352.36280219444086), (936, 1829, 1300.8794642857142, 361.9975319663285), (928, 1786, 1240.4913294797689, 384.0008974353842), (907, 1745, 1431.8704225352112, 333.1159593364002), (887, 1707, 1399.53, 242.41932493099637), (868, 1670, 1225.8206896551724, 374.1837584518862), (849, 1634, 1381.383527454243, 164.5969063871371), (832, 1600, 1181.8403361344538, 342.73231054618117), (815, 1567, 1339.3428571428572, 248.49359783154435), (798, 1536, 1083.0681818181818, 211.99604924784379), (783, 1506, 1081.326923076923, 251.95727821275315), (768, 1477, 1189.5072463768115, 285.8600358434483), (753, 1449, 1118.2432432432433, 281.96208345875215), (739, 1422, 1214.0212765957447, 223.4919665710036), (726, 1396, 1114.3181818181818, 184.1267193004062), (713, 1371, 1091.1818181818182, 311.46831684249076), (700, 1347, 1112.450127877238, 197.745999325762), (688, 1324, 1049.7777777777778, 218.2675204282918), (677, 1302, 1169.8510638297873, 209.69988471099464), (607, 1280, 937.0887825624668, 255.27686148416126)]
    height_per_zoom_dist = [(24160, 48000, 34582.545454545456, 4122.684129272102), (9853, 40846, 16490.03961709244, 2221.30538133036), (6569, 20375, 11500.962139917696, 1584.9441125426117), (5483, 15669, 8435.98998487745, 1119.8328788922317), (3941, 12225, 6753.860526605707, 1020.0673029608719), (3284, 10188, 5823.7535916938305, 787.7561927128141), (2815, 8732, 4981.578917636425, 791.2456956889796), (2463, 7641, 4338.460869565218, 576.4211152038307), (2190, 6792, 3912.547923942569, 502.67489653815693), (1971, 7240, 3323.9517137765974, 446.84795365074393), (1791, 4364, 3132.636927744071, 457.4231728389159), (1642, 8399, 2951.702337369755, 431.6551565078864), (1516, 4374, 2716.8520739630185, 418.1395828566948), (1483, 4062, 2543.7697551123456, 389.7904252498075), (1611, 3791, 2318.4827664916684, 349.7909128453299), (1232, 3554, 2215.294685990338, 307.78949472577636), (1159, 3345, 2130.2518325845354, 318.4048036873566), (1095, 3159, 2009.0177705977383, 316.5059578144224), (1093, 3597, 1938.708867978329, 306.47103535274147), (985, 2880, 1674.2580576609762, 234.12260419871743), (989, 2708, 1663.2148148148149, 292.13344579299996), (896, 2585, 1639.8813236972233, 315.78639964211965), (857, 2472, 1507.3540983606558, 276.31452988052246), (821, 2369, 1465.4586872586872, 240.8099549305292), (788, 2275, 1461.1454154727794, 253.49360305795298), (978, 2187, 1336.7565217391304, 211.60572817529624), (895, 1886, 1391.6675555555555, 234.06721478294844), (704, 2031, 1290.048231511254, 196.6726414598678), (833, 1961, 1211.8478824546241, 194.62459315860843), (848, 1697, 1236.447342995169, 203.13324886009488), (636, 1643, 1112.3836898395723, 205.87308821348213), (616, 1591, 1106.581081081081, 154.27157434564384), (771, 1543, 1118.3959484346224, 159.3933473306579), (748, 1447, 1114.1818181818182, 177.55316616933902), (563, 1625, 1053.5856697819315, 196.77656173019756), (547, 1333, 1017.546, 107.37988584460312), (688, 1376, 1094.042372881356, 154.11953285657455), (519, 1263, 925.1884272997032, 110.12831112199531), (652, 1306, 886.6735537190083, 158.4254986165074), (493, 1440, 866.8214359969506, 129.2520341201543), (620, 1171, 890.0155440414508, 138.64899494707328), (606, 1143, 869.53125, 141.822005225898), (592, 1116, 790.1398843930635, 125.21977938456682), (578, 1091, 867.1830985915493, 115.81083530264912), (565, 1067, 863.065, 119.74519103078839), (553, 1043, 768.7333333333333, 89.19709632234205), (541, 1021, 872.1364392678869, 66.40820434223562), (530, 1000, 731.7142857142857, 123.41289193904625), (519, 980, 756.2285714285714, 86.14633255249407), (526, 960, 725.8125, 66.70273320095117), (569, 941, 686.0, 89.07635013765525), (506, 923, 713.2536231884058, 88.8856178837873), (480, 906, 688.9594594594595, 95.66713677730338), (551, 889, 695.5212765957447, 62.961948779520654), (527, 873, 674.3636363636364, 63.947986999766556), (532, 857, 649.0727272727273, 70.14016820333615), (522, 842, 691.1508951406649, 70.8341259163928), (500, 828, 672.0322580645161, 78.49705402041037), (492, 814, 652.6808510638298, 72.89517579781833), (424, 849, 587.9290271132377, 83.53495235100546)] 
    width = np.random.normal(width_per_zoom_dist[zoom-1][2], width_per_zoom_dist[zoom-1][3])
    width = int(np.clip(width, width_per_zoom_dist[zoom-1][0], width_per_zoom_dist[zoom-1][1]))
    height = np.random.normal(height_per_zoom_dist[zoom-1][2], height_per_zoom_dist[zoom-1][3])
    height = int(np.clip(height, height_per_zoom_dist[zoom-1][0], height_per_zoom_dist[zoom-1][1]))
    return width, height #W, H (40X scale)

def random_patch_number_generator(zoom):
    patch_number_min = 69.66666666666667
    patch_number_max = 709.125
    patch_number_mean = 278.47684144818976
    patch_number_std = 131.63679609746254
    # patch_number_to_zoom_lvl_probabilities = [0.05734461977273773, 0.19217397590678173, 0.31772027944452846, 0.27394698966212977, 0.13471763089925667, 0.024096504314565684]
    # bin (0,2)->1 (3,6)->5 (7,12)->10 (13-27)->20 (28 45)->40 (>)50
    patch_number_to_zoom_lvl_probabilities = {1:0.05734461977273773, 5:0.19217397590678173, 10:0.31772027944452846, 20:0.27394698966212977, 40:0.13471763089925667, 50:0.024096504314565684}
    patch_number = np.random.normal(patch_number_mean, patch_number_std)
    patch_number = int(np.clip(patch_number, patch_number_min, patch_number_max))
    num_patches_per_zoom = patch_number*patch_number_to_zoom_lvl_probabilities[zoom] 
    return num_patches_per_zoom

def translate_encoder(zoom, X, Y):
    x= X//zoom
    y= Y//zoom
    return x, y 

def translate_decoder(zoom, x, y):
    X= x*zoom
    Y= y*zoom
    return X, Y 

def random_xy_fgbg_generator(image):
    segmented_image = segment_background_foreground(image)
    fg_bg= segmented_image[:,:,0] 
    non_black_coordinates = np.argwhere(fg_bg != 0)

    # Randomly select a coordinate from the non-black regions
    if len(non_black_coordinates) > 0:
        random_coordinate = random.choice(non_black_coordinates)
        pixel_x, pixel_y = random_coordinate[0], random_coordinate[1]
    else:
        # Handle the case where there are no non-black regions found
        # This could happen if the segmentation does not identify any non-black regions
        # You may want to take appropriate action here, such as choosing a fallback strategy
        pixel_x, pixel_y = None, None

    return pixel_x,pixel_y,fg_bg

def calculate_overlap(rect1, rect2):
    # Calculate the overlap area between two rectangles
    x_overlap = max(0, min(rect1[0]+rect1[2], rect2[0]+rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[1]+rect1[3], rect2[1]+rect2[3]) - max(rect1[1], rect2[1]))
    overlap_area = x_overlap * y_overlap
    return overlap_area

def delete_helper(new_data, value, threshold):
    overlapping_helper = False        
    for i in range(len(new_data)):
        overlap_area = calculate_overlap(new_data[i][1:5], value[1:5])
        total_area = new_data[i][3] * new_data[i][4]
        if overlap_area / total_area > threshold:
            overlapping_helper = True           
    return overlapping_helper

def delete_overlapping_rectangles(data, threshold=0.3): #threshold -> allowed overlap percentage
    new_data = []
    for i in range(len(data)):
        overlapping = False
        for j in range(len(data)):
            if i != j:
                overlap_area = calculate_overlap(data[i][1:5], data[j][1:5])
                total_area = data[i][3] * data[i][4]
                if overlap_area / total_area > threshold:
                    overlapping = True
                    break
        if not overlapping:
            new_data.append(data[i])
        else:
            overlapping_helper = delete_helper(new_data, data[i], threshold)
            if not overlapping_helper:
                new_data.append(data[i])
    return new_data

def overlap_remover(temp_result):
    residue = 0
    temp_cut_result = delete_overlapping_rectangles(temp_result)
    residue = len(temp_result) - len(temp_cut_result)
    return temp_cut_result,residue 

def fg_bg_area(x,y,w,h,fg_bg):
    roi = fg_bg[y:y+h, x:x+w]
    non_black_pixels = cv2.countNonZero(roi)

    # Calculate the total number of pixels in the region
    total_pixels = w * h

    # Check if more than 50% of pixels are non-black
    if non_black_pixels > total_pixels / 2:
        return False
    else:
        return True
  
def master(folder_path, csv_file_path):
    zoom_dict = [1,5,10,20,40,50]

    # Get a list of all files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.endswith(('.tif'))]

    # Iterate over each image file
    for image_file in image_files:
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)    
        image = tifi.imread(image_path)
        result = []
        residue = 0

        for zoom in zoom_dict:
            patch_number = int(random_patch_number_generator(zoom)+residue)
            temp_result = []
            #block 1 
            for i in range(patch_number):
                zoom_image = zoom_in(image,zoom)
                #block 2
                x, y, fg_bg = random_xy_fgbg_generator(zoom_image)
                W,H = random_WH_generator(zoom)
                w,h = translate_encoder(zoom, W, H)
                if(fg_bg_area(x,y,w,h,fg_bg)):
                    continue
                #Edges
                max_x, max_y = zoom_image[:,:,0].shape
                if((x+w<max_x) and (y+h<max_y)):
                    X,Y = translate_decoder(zoom, x, y)
                    W,H = translate_decoder(zoom, w, h)
                    file_name = os.path.splitext(image_file)[0]
                    temp_result.append([file_name, X, Y, W, H, zoom])
            if(len(temp_result)>1):
                temp_result,residue = overlap_remover(temp_result)
            result.extend(temp_result)

        break#To run for single image




    # Open the CSV file in append mode
    with open(csv_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow(["File_name","x","y","w","h","zoom"])
        # Append the data to the CSV file
        for row in result:
            writer.writerow(row)

    print("Data appended successfully.")


# Path to the folder containing images
#C:/Users/stlp/Desktop/Linda/40X
folder_path = "C:/Users/stlp/Desktop/Linda/40X"
csv_file_path = "C:/Users/stlp/Desktop/Linda/data_gen_final.csv"
master(folder_path, csv_file_path) 
