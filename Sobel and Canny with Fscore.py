#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

def sobel_edge_detection(image_path):
    """
    Performs edge detection on the given image using the Sobel edge detection algorithm.

    Args:
        image_path: The path of the image file.

    Returns:
        The edge-detected image using the Sobel algorithm.
    """
    image = cv2.imread(image_path, 0)  # Load the image as grayscale

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of gradients
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return magnitude

def canny_edge_detection(image_path):
    """
    Performs edge detection on the given image using the Canny edge detection algorithm.

    Args:
        image_path: The path of the image file.

    Returns:
        The edge-detected image using the Canny algorithm.
    """
    image = cv2.imread(image_path, 0)  # Load the image as grayscale

    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    return edges

def calculate_f_score(ground_truth, detected_edges):
    """
    Calculates the F-score for the detected edges compared to the ground truth.

    Args:
        ground_truth: The ground truth image containing the correct edge pixels.
        detected_edges: The edge-detected image obtained from the algorithm.

    Returns:
        The F-score.
    """
    # Threshold the ground truth and detected edges to obtain binary images
    _, ground_truth_threshold = cv2.threshold(ground_truth, 128, 255, cv2.THRESH_BINARY)
    _, detected_edges_threshold = cv2.threshold(detected_edges, 128, 255, cv2.THRESH_BINARY)

    # Calculate true positives, false positives, and false negatives
    true_positives = np.count_nonzero(np.logical_and(ground_truth_threshold, detected_edges_threshold))
    false_positives = np.count_nonzero(np.logical_and(255 - ground_truth_threshold, detected_edges_threshold))
    false_negatives = np.count_nonzero(np.logical_and(ground_truth_threshold, 255 - detected_edges_threshold))

    # Calculate precision, recall, and F-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 * (precision * recall) / (precision + recall)

    return f_score

# Example usage
image_path = "fish.png"
ground_truth_path = "fish.png"

# Load the original image and the ground truth image
original_image = cv2.imread(image_path)

# Perform edge detection using Sobel
sobel_edges = sobel_edge_detection(image_path)

# Perform edge detection using Canny
canny_edges = canny_edge_detection(image_path)

# Load the ground truth image
ground_truth_image = cv2.imread(ground_truth_path, 0)

# Calculate the F-score for Sobel
sobel_f_score = calculate_f_score(ground_truth_image, sobel_edges)

# Calculate the F-score for Canny
canny_f_score = calculate_f_score(ground_truth_image, canny_edges)

print("Sobel F-score:", sobel_f_score)
print("Canny F-score:", canny_f_score)

# Display the original image, Sobel edges, and Canny edges
cv2.imshow("Original Image", original_image)
cv2.imshow("Sobel Edge-Detected Image", sobel_edges)
cv2.imshow("Canny Edge-Detected Image", canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




