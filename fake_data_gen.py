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
    width_per_zoom_dist = [(0.3326612903225806, 0.967741935483871, 0.47051207018485985, 0.1757260367902772), (0.10556675627240143, 0.9778225806451613, 0.39250942833036556, 0.19469698912435957), (0.07037550403225806, 0.9408602150537635, 0.3431542997281968, 0.1687248793625112), (0.052783378136200716, 0.967741935483871, 0.28009296240148174, 0.13347355306200748), (0.04222670250896057, 0.7741935483870968, 0.23681644904186183, 0.12274217353962108), (0.03519125224014337, 0.8433719758064516, 0.20963372573546574, 0.09862939068567453), (0.030164930555555556, 0.7229082661290323, 0.1734817806221589, 0.09058521294735589), (0.027323853989813244, 0.8064516129032258, 0.166060703606124, 0.07950696347168448), (0.023458501344086023, 0.716817876344086, 0.1517173541355015, 0.06983463428555138), (0.021113351254480286, 0.967741935483871, 0.12648836856436035, 0.06089855478670249), (0.019195228494623656, 0.5865255376344086, 0.12352006524074728, 0.06786372418177938), (0.017592125896057347, 0.8064516129032258, 0.11463836598983698, 0.061879462054314924), (0.016241039426523298, 0.38923891129032256, 0.10894654358524304, 0.06055022149396877), (0.015078965053763441, 0.6912802419354839, 0.10046216274598899, 0.05407285339936581), (0.014077900985663083, 0.6451612903225806, 0.09295740638415759, 0.05473113208369129), (0.013195844534050179, 0.6048387096774194, 0.08888128267674825, 0.05042382252107897), (0.015571943972835314, 0.29763104838709675, 0.08694282627003999, 0.03995564708151734), (0.014709783531409167, 0.2811239919354839, 0.08340766455815973, 0.04310488716102491), (0.013933839134125636, 0.5093245967741935, 0.08071362414051135, 0.0443683897855539), (0.012651209677419355, 0.4838709677419355, 0.06491547695205555, 0.03327939184536718), (0.01260743845500849, 0.4608114919354839, 0.07337530473395831, 0.05245549840865619), (0.012037086162988115, 0.22996471774193547, 0.06940250282165887, 0.03376353336549007), (0.011513157894736841, 0.28049395161290325, 0.05555498740308195, 0.0315423710765094), (0.011029021646859084, 0.26881720430107525, 0.06162690810170151, 0.03535560473380483), (0.010591309422750425, 0.3870967741935484, 0.06278995936173304, 0.04510683519557428), (0.009690020161290322, 0.3722278225806452, 0.06056750019257398, 0.0355267323396706), (0.009330897177419355, 0.35836693548387094, 0.06797292844178165, 0.04922028983403617), (0.009457236842105263, 0.34564012096774194, 0.054699389427697385, 0.035038150971058564), (0.009132268675721563, 0.17452116935483872, 0.06119078946284813, 0.03874566313031502), (0.008827196519524618, 0.16872479838709678, 0.056916270233792844, 0.036830904663086354), (0.008542020373514432, 0.1631804435483871, 0.048882624391452814, 0.032893784358165624), (0.008270108234295416, 0.3024193548387097, 0.04299884396073107, 0.024868318865156336), (0.008024724108658744, 0.29322076612903225, 0.04990369329222517, 0.02780514198562271), (0.007785971986417657, 0.2846522177419355, 0.04815694131340656, 0.02655397678151357), (0.0075671158743633275, 0.14453125, 0.055177075112483424, 0.04063265394081346), (0.007354891765704584, 0.26877520161290325, 0.03817299937407079, 0.01427629316563891), (0.007155931663837012, 0.10463709677419354, 0.05684959773191897, 0.02314087165208741), (0.0069702355687606115, 0.10186491935483871, 0.04262168268086277, 0.016087103444128944), (0.006791171477079796, 0.1297883064516129, 0.042267934807875315, 0.02816108382163993), (0.0063256048387096775, 0.24193548387096775, 0.029238975868471864, 0.015791074720339953), (0.00645957130730051, 0.2360131048387097, 0.04465100212838252, 0.024356721139380626), (0.006300403225806451, 0.23046875, 0.03945068202331935, 0.02872455744033419), (0.006154499151103565, 0.11769153225806452, 0.038686543282833726, 0.025580730228655594), (0.006015227079796265, 0.11504536290322581, 0.042265578339165534, 0.019518788196905765), (0.00588258701188455, 0.08603830645161291, 0.0371484517057679, 0.013604271449681512), (0.005756578947368421, 0.08417338709677419, 0.04582708675652316, 0.012767204528815867), (0.005630570882852292, 0.20589717741935484, 0.03523056146358592, 0.007117234321535519), (0.0055178268251273345, 0.08064516129032258, 0.03446089235804949, 0.019651665747490568), (0.005405082767402377, 0.07898185483870968, 0.042755031608814206, 0.017529923953741578), (0.007602486559139785, 0.07741935483870968, 0.024718103819737915, 0.013086865988342446), (0.007113155241935484, 0.1897681451612903, 0.03247561841435558, 0.019188652866447047), (0.006980846774193548, 0.1861139112903226, 0.03611342385546816, 0.01713526793096463), (0.004993898556876061, 0.18258568548387097, 0.037335845903731096, 0.024959109367165947), (0.009272455506117909, 0.07167338709677419, 0.04105178603748652, 0.021992910684355292), (0.006596522177419355, 0.07036290322580645, 0.028988199143675418, 0.010893262724449165), (0.012966229838709677, 0.06910282258064517, 0.042021011128988595, 0.020686380873298212), (0.0063697076612903225, 0.06789314516129032, 0.0328499485878577, 0.014383663309645177), (0.008780772495755517, 0.06673387096774193, 0.02973409349769338, 0.007649003224221521), (0.008481646273637376, 0.065625, 0.03263850584233744, 0.013674905678223152), (0.0044102822580645165, 0.16129032258064516, 0.02367988648848098, 0.012612746250591628)]
    height_per_zoom_dist = [(0.31005859375, 0.7931926751592356, 0.5345556915450416, 0.09317342920253363), (0.13916666666666666, 0.9814453125, 0.3184589288602926, 0.12493036380373097), (0.09657728580562659, 0.9103260869565217, 0.2677494774991636, 0.11057849378216578), (0.06958333333333333, 0.8488175675675675, 0.21585817970571733, 0.09181252525167632), (0.06201171875, 0.7211538461538461, 0.1807955030855744, 0.07857762109425495), (0.050341796875, 0.717391304347826, 0.15530376469651888, 0.06788708367482472), (0.044291178385416664, 0.5150991586538461, 0.13480540497446475, 0.05908929901263597), (0.03875732421875, 0.6334459459459459, 0.12699617300288366, 0.055671891998584785), (0.034454345703125, 0.5630278716216216, 0.11084437995039673, 0.04828096716046241), (0.027833333333333335, 0.5067567567567568, 0.09742501125693372, 0.044716191460987334), (0.028188069661458332, 0.46072635135135137, 0.09104757588176468, 0.04068493563231589), (0.025838216145833332, 0.4222972972972973, 0.08531930362755265, 0.04138682620292187), (0.023854573567708332, 0.31351902173913043, 0.08022866202822206, 0.038359840857842234), (0.022145589192708332, 0.3620143581081081, 0.07703261240112139, 0.03884856376123681), (0.020670572916666668, 0.33783783783783783, 0.0688660958940707, 0.03593400194274145), (0.019378662109375, 0.31672297297297297, 0.0662963200735018, 0.0323595351741315), (0.018239339192708332, 0.23980978260869565, 0.06461869019155575, 0.0283162556498341), (0.017222086588541668, 0.22647758152173914, 0.06054543717219418, 0.03153685462421935), (0.016316731770833332, 0.26668074324324326, 0.058586550585680325, 0.03093015399111421), (0.013916666666666667, 0.2533783783783784, 0.0497428160551368, 0.024304731933783887), (0.015295316496163683, 0.2413429054054054, 0.05378828836122634, 0.030674829443707162), (0.014088948567708334, 0.1639122596153846, 0.05184938443940648, 0.02386418262733422), (0.013478597005208334, 0.22033361486486486, 0.043425698447309515, 0.02124499147184711), (0.012919108072916666, 0.21114864864864866, 0.04431464617787083, 0.021335550823006363), (0.012847666240409207, 0.17857142857142858, 0.04518540222053704, 0.02557517499767744), (0.011922200520833334, 0.17168898809523808, 0.0459010252437867, 0.023428456116751137), (0.0118985773657289, 0.18771114864864866, 0.049970150160818294, 0.03633985592014409), (0.011077880859375, 0.15941220238095238, 0.03992817459423388, 0.02373912125391679), (0.010691324869791666, 0.14054008152173914, 0.04513745371990738, 0.024432285000398672), (0.010709718670076727, 0.1358695652173913, 0.039843809878416424, 0.023051168124289298), (0.011598865089514066, 0.09644717261904762, 0.034140889063469806, 0.018255273403915414), (0.010040361253196932, 0.13950892857142858, 0.03312756852054201, 0.015268393513577084), (0.0093994140625, 0.13532366071428573, 0.034906414653817254, 0.015158878187024731), (0.009221147698209718, 0.13132440476190477, 0.03502905962258203, 0.01840221509941268), (0.009181186061381075, 0.11642323369565218, 0.04412524884327341, 0.03363562580753337), (0.008921435421994885, 0.12397693452380952, 0.01981671439706034, 0.012077290102116247), (0.008681665601023018, 0.09743088942307693, 0.03739096053213883, 0.014463965317640971), (0.00825207800511509, 0.09487680288461539, 0.03378360953670738, 0.015327439858018198), (0.008242087595907928, 0.10453464673913043, 0.02704665033337627, 0.014408426683654161), (0.007293701171875, 0.11160714285714286, 0.02227062792536555, 0.01189853133115923), (0.007832480818414322, 0.10890997023809523, 0.03138259925696894, 0.015060489825869966), (0.00746283567774936, 0.10630580357142858, 0.03020242396253959, 0.01718248460544753), (0.007292998721227621, 0.09476902173913043, 0.025033925024244074, 0.010117612900575093), (0.0081721547314578, 0.06798735119047619, 0.02926635927868777, 0.009529272964322197), (0.0079923273657289, 0.0546875, 0.026967968631580623, 0.008465465639783216), (0.008114583333333333, 0.053498641304347824, 0.03287110720760701, 0.015400080140222363), (0.00848707329317269, 0.09495907738095238, 0.026418158135003696, 0.003704016254865233), (0.00778125, 0.05010516826923077, 0.022866122915288233, 0.009976150936111794), (0.008141942771084338, 0.04845727848101266, 0.023507722310759566, 0.008000424577377983), (0.009005681818181819, 0.049252717391304345, 0.02009179198715274, 0.006228413793243929), (0.0086181640625, 0.08751860119047619, 0.02209370095176016, 0.009463039040930191), (0.008447265625, 0.08584449404761904, 0.026879075172561273, 0.008567247802872047), (0.006474905303030303, 0.08426339285714286, 0.025245110702424002, 0.013100175256982169), (0.009758309925093633, 0.04071044921875, 0.02502713526276764, 0.00913704581592485), (0.00799560546875, 0.044752038043478264, 0.0193273576865144, 0.007800496743198784), (0.009514086174242424, 0.04237539556962025, 0.023546787281798497, 0.0047248803918377245), (0.00771484375, 0.035310444078947366, 0.022856899694239017, 0.005493196483937597), (0.009860899390243903, 0.042459239130434784, 0.019986200015293592, 0.002998391063169877), (0.006322916666666667, 0.041779891304347824, 0.021923505878853364, 0.007762217984182593), (0.005205003196930946, 0.0744047619047619, 0.017065887271729863, 0.00638432693665847)]
    width = np.random.normal(width_per_zoom_dist[zoom-1][2], width_per_zoom_dist[zoom-1][3])
    width = np.clip(width, width_per_zoom_dist[zoom-1][0], width_per_zoom_dist[zoom-1][1])
    height = np.random.normal(height_per_zoom_dist[zoom-1][2], height_per_zoom_dist[zoom-1][3])
    height = np.clip(height, height_per_zoom_dist[zoom-1][0], height_per_zoom_dist[zoom-1][1])
    return width, height #W, H ratios = crop w,h / image w, h

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

def translate_encoder(max_x, max_y, W, H):
    w= max_x*W
    h= max_x*H
    return w, h 

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
        return True
    else:
        return False
  
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
            print(patch_number)
            temp_result = []
            #block 1 
            i= 0
            max_retry = 20
            max_retry_edges = 5
            while i<patch_number:
                zoom_image = zoom_in(image,zoom)
                #block 2
                x, y, fg_bg = random_xy_fgbg_generator(zoom_image)
                max_x, max_y = zoom_image[:,:,0].shape
                W,H = random_WH_generator(zoom)
                w,h = translate_encoder(max_x, max_y, W, H)
                w= int(w)
                h= int(h)
                print(x,y,w,h)
                if((not fg_bg_area(x,y,w,h,fg_bg)) and max_retry>0):
                    max_retry -= 1
                    continue
                #Edges
                if((x+w<max_x) and (y+h<max_y)):
                    X,Y = translate_decoder(zoom, x, y)
                    W,H = translate_decoder(zoom, w, h)
                    file_name = os.path.splitext(image_file)[0]
                    temp_result.append([file_name, X, Y, W, H, zoom])
                else:
                    if(max_retry_edges>0):
                        max_retry_edges -=1
                        continue 
                max_retry_edges = 5   
                i = i+1    
            if(len(temp_result)>1):
                temp_result,residue = overlap_remover(temp_result)
            result.extend(temp_result)
            break    

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
#projects/brain1/tejoram/Navigation
folder_path = "40X"
csv_file_path = "data_gen_final.csv"
master(folder_path, csv_file_path) 
