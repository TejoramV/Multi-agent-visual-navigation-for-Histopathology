import pandas as pd
import numpy as np
from glob import glob
import cv2 
import random
import csv
import tifffile as tifi #pip install imagecodecs
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    ll =200
    lower = np.array([ll, ll, ll], dtype="uint8")
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

def translate_to_overlap(bbox):
    translated_bbox = []
    for i in bbox:
        translated_bbox.append([-1, i[0], i[1], i[2], i[3], -1])
    return translated_bbox

def replace_overlapping(rectangles):
    # Sort the rectangles by their area in descending order
    rectangles.sort(key=lambda rect: rect[3] * rect[4], reverse=True)

    new_rectangles = []
    for rect in rectangles:
        x1, y1, w1, h1 = rect[1], rect[2], rect[3], rect[4]
        overlapping = False
        for i, other_rect in enumerate(new_rectangles):
            x2, y2, w2, h2 = other_rect[1], other_rect[2], other_rect[3], other_rect[4]
            # Check for overlap
            if (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                # Overlapping, create a bigger rectangle and replace the overlapping rectangle
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                new_rectangles[i] = (-1, new_x, new_y, new_w, new_h, -1)
                overlapping = True
                break
        if not overlapping:
            new_rectangles.append(rect)
    return new_rectangles

def merge_bounding_boxes(bounding_boxes):
    # Sort the bounding boxes by their x1 coordinate (the first element in each bounding box)
    bounding_boxes.sort(key=lambda box: box[0])

    # Initialize the list of merged bounding boxes
    merged_boxes = []

    # Iterate through each bounding box
    for box in bounding_boxes:
        if not merged_boxes:
            merged_boxes.append(box)
        else:
            # Get the last box in the merged list
            last_box = merged_boxes[-1]

            # Check if there is an overlap in the x-values
            if box[0] <= last_box[2]:
                # If they overlap, merge the bounding boxes
                new_box = [
                    min(last_box[0], box[0]),  # x1
                    min(last_box[1], box[1]),  # y1
                    max(last_box[2], box[2]),  # x2
                    max(last_box[3], box[3])   # y2
                ]
                merged_boxes[-1] = new_box
            else:
                # If they don't overlap, add the current box to the merged list
                merged_boxes.append(box)

    return merged_boxes

def sort_bboxes(bboxes, row_threshold=100):
    """
    Sorts the bounding boxes by y-coordinate and then by x-coordinate within each row.
    
    Args:
    - bboxes (list of tuples): List of bounding boxes of the tissue segments in the format (x1, y1, x2, y2).
    - row_threshold (int): The threshold to consider segments in the same row.
    
    Returns:
    - sorted_bboxes (list of tuples): Sorted bounding boxes.
    """
    # Sort bboxes by y-coordinate
    bboxes = sorted(bboxes, key=lambda x: (x[1], x[0]))  # Sort by y1 first and then x1
    
    rows = []
    current_row = [bboxes[0]]
    
    for bbox in bboxes[1:]:
        if abs(bbox[1] - current_row[-1][1]) <= row_threshold:
            current_row.append(bbox)
        else:
            rows.append(current_row)
            current_row = [bbox]
    
    if current_row:
        rows.append(current_row)
    
    # Sort each row by x-coordinate
    sorted_bboxes = []
    for row in rows:
        sorted_bboxes.extend(sorted(row, key=lambda x: x[0]))  # Sort by x1 within each row
    
    return sorted_bboxes

def group_segments_into_rows(bboxes, threshold=100):
    """
    Groups segments into rows based on their y-coordinates.
    
    Args:
    - bboxes (list of tuples): List of bounding boxes of the tissue segments in the format (x1, y1, x2, y2).
    - threshold (int): The threshold to consider segments in the same row.
    
    Returns:
    - rows (list of lists): Grouped bounding boxes into rows.
    """
    # bboxes = sorted(bboxes, key=lambda x: x[1])  # Sort by y1
    rows = []
    current_row = [bboxes[0]]
    
    for bbox in bboxes[1:]:
        if abs(bbox[1] - current_row[-1][1]) <= threshold:
            current_row.append(bbox)
        else:
            rows.append(current_row)
            current_row = [bbox]
    
    if current_row:
        rows.append(current_row)
    
    return rows

def stitch_segments_with_margin(image, bboxes, random_pixel, margin=50,threshold=100 ):
    """
    Stitches tissue segments together with minimal white background, maintaining their relative positions and adding a margin.
    
    Args:
    - image (np.array): The original histopathology image.
    - bboxes (list of tuples): List of bounding boxes of the tissue segments in the format (x1, y1, x2, y2).
    - margin (int): Margin to add between segments.
    
    Returns:
    - stitched_image (np.array): The stitched image with minimal white background and added margin.
    """
    # # Sort the bounding boxes by their x and y coordinates
    bboxes = sort_bboxes(bboxes)
    
    # Extract segments from the image
    segments = [(image[y1:y2, x1:x2], x1, y1, x2, y2) for x1, y1, x2, y2 in bboxes]
    
    # Group segments into rows
    rows = group_segments_into_rows(bboxes, threshold)
    
    # Determine the dimensions of the stitched image
    max_row_height = [max([y2 - y1 for x1, y1, x2, y2 in row]) for row in rows]
    total_height = sum(max_row_height) + (len(rows) + 1) * margin
    max_row_width = [sum([x2 - x1 for x1, y1, x2, y2 in row]) for row in rows]
    total_width = max(max_row_width) + margin * (max(len(row) for row in rows) + 1)

    # Create a blank canvas with bacground #white background
    stitched_image = np.ones((total_height, total_width, image.shape[2]), dtype=np.uint8) * image[random_pixel[0],random_pixel[1]] #255
    
    current_y = margin
    for row in rows:
        current_x = margin
        row_height = max([y2 - y1 for x1, y1, x2, y2 in row])
        for x1, y1, x2, y2 in row:
            segment = image[y1:y2, x1:x2]
            h, w = segment.shape[:2]
            stitched_image[current_y:current_y+h, current_x:current_x+w] = segment
            current_x += w + margin
        current_y += row_height + margin
    
    return stitched_image

def main(image_path):
    image = tifi.imread(image_path)
    bw = segment_background_foreground(image)
    bw = bw[:,:,0]
    _, thresh = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)

    background_coords = np.column_stack(np.where(thresh < 50))

    # Check if there are any background pixels found
    if background_coords.size == 0:
        raise ValueError("No background pixels found with value less than 50")

    # Randomly select one of these coordinates
    random_index = np.random.randint(0, len(background_coords))
    random_pixel = background_coords[random_index]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    area = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w>2 and h>2: #To speed up the code, works the same without this hardcoding
            bbox.append([x, y, w, h])
            area.append(w*h)
    # if area is less than 1000, remove the bounding box
    bbox = [i for i in bbox if (i[2]*i[3]>1000 )]
    bbox = [i for i in bbox if (i[2]<10*i[3])] #delete edge feature
    translated_bbox = translate_to_overlap(bbox)
    temp_cut_result = replace_overlapping(translated_bbox)
    bounding_boxes =[]
    for index,value in enumerate(temp_cut_result):
        x,y,w,h = value[1],value[2],value[3],value[4]
        # cv2.rectangle(temp_img3, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue bounding boxes
        bounding_boxes.append([x,y,x + w,y + h])
    merged_boxes = merge_bounding_boxes(bounding_boxes)
    bounding_boxes = merged_boxes
    x_min_all = min(box[0] for box in bounding_boxes)
    y_min_all = min(box[1] for box in bounding_boxes)
    x_max_all = max(box[2] for box in bounding_boxes)
    y_max_all = max(box[3] for box in bounding_boxes)
    for box in bounding_boxes:
        box[1]= y_min_all
        box[-1]=y_max_all

    stitched_width = x_max_all - x_min_all
    stitched_height = y_max_all - y_min_all
    stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        cropped_box = image[y_min:y_max, x_min:x_max]
        
        # Adjust coordinates for pasting onto stitched image
        x_offset = x_min - x_min_all
        y_offset = y_min - y_min_all
        cropped_box = cv2.cvtColor(cropped_box, cv2.COLOR_RGBA2BGR)  # or cv2.COLOR_RGBA2BGR
        
        # Paste the cropped bounding box onto the stitched image
        stitched_image[y_offset:y_offset + (y_max - y_min), x_offset:x_offset + (x_max - x_min)] = cropped_box
    sorted_bboxes = sort_bboxes(bounding_boxes)
    stitched_output_image= stitch_segments_with_margin(image, bounding_boxes, random_pixel, margin=200, threshold=1000)
    return stitched_output_image


dataset_path = "/projects/melanoma/mpath2/tiff"

for folder in tqdm(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for subfolder in os.listdir(folder_path):
            try:
                img_folder = os.path.join(folder_path, subfolder, "*_level1.tiff")
                stitched_output_image = main(img_folder)
                level1_file = glob.glob(img_folder)
                filename = os.path.basename(level1_file)
                output_file_name = filename.replace('_level1.tiff', '_level1_processed.tiff')
                output_path = os.path.join(folder_path, subfolder,output_file_name)
                cv2.imwrite(output_path, stitched_output_image)
            except:
                continue


