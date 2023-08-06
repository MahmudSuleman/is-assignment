#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import random

# BPSO Parameters
population_size = 20
iterations = 50
max_threshold = 255
# code to find the BPSO
def bpso_edge_detection(image_path):
    """
    Performs edge detection on the given image using BPSO.

    Args:
        image_path: The path of the image file.

    Returns:
        The edge-detected image.
    """
    image = cv2.imread(image_path, 0)  # Load the image as grayscale

    # Initialize the swarm of particles
    particles = []
    for _ in range(population_size):
        particle = [random.randint(0, max_threshold), float('-inf')]
        particles.append(particle)

    # Perform BPSO optimization
    global_best_particle = None
    global_best_fitness = float('-inf')

    for _ in range(iterations):
        for particle in particles:
            threshold = particle[0]

            # Apply thresholding to the image
            _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

            # Calculate fitness based on the number of edge pixels
            fitness = np.count_nonzero(thresholded_image)

            # Update personal best
            if fitness > particle[1]:
                particle[1] = fitness

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_particle = particle

    # Apply thresholding with the global best threshold
    best_threshold = global_best_particle[0]
    _, edge_detected_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)

    return edge_detected_image

# Example usage
image_path = "fish.png"
edge_detected_image = bpso_edge_detection(image_path)

# Display the original and edge-detected images
cv2.imshow("Original Image", cv2.imread(image_path))
cv2.imshow("Edge-Detected Image", edge_detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




