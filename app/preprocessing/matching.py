import cv2
import numpy as np
import argparse
from sklearn.cluster import KMeans
from scipy.spatial import distance
import sys

def load_image(image_path):
    """Loads an image and converts it to LAB color space."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert to LAB space
    return image

def apply_mask(image, mask_path):
    """Applies a mask to extract the clothing region from the person image."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def extract_dominant_colors(image, k=3):
    """Extracts dominant colors using K-Means clustering and returns sorted cluster centers."""
    pixels = image.reshape(-1, 3)
    pixels = pixels[np.any(pixels > 0, axis=1)]  # Remove black (masked-out) pixels
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    cluster_centers = kmeans.cluster_centers_
    return sorted(cluster_centers, key=lambda x: np.linalg.norm(x))

def compute_color_histogram(image):
    """Computes normalized color histogram in LAB space."""
    hist = []
    for i in range(3):
        h = cv2.calcHist([image], [i], None, [256], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist.append(h)
    return np.concatenate(hist)

def compare_histograms(hist1, hist2):
    """Compares histograms using Bhattacharyya distance."""
    return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)

def compare_colors(color_list1, color_list2):
    """Computes minimum Euclidean distance between dominant colors of two images."""
    return min(distance.euclidean(c1, c2) for c1 in color_list1 for c2 in color_list2)

def match_clothing(image_cloth, image_person, mask_path, color_threshold=15, hist_threshold=0.5):
    """Matches clothing based on dominant color and histogram comparison using a mask."""
    # Load images
    cloth_lab = load_image(image_cloth)
    person_lab = load_image(image_person)
    masked_person_lab = apply_mask(person_lab, mask_path)
    
    # Extract dominant colors
    colors_cloth = extract_dominant_colors(cloth_lab)
    colors_person = extract_dominant_colors(masked_person_lab)
    
    # Compute histograms
    hist_cloth = compute_color_histogram(cloth_lab)
    hist_person = compute_color_histogram(masked_person_lab)
    
    # Compare color distance
    color_diff = compare_colors(colors_cloth, colors_person)
    
    # Compare histogram similarity
    hist_diff = compare_histograms(hist_cloth, hist_person)
    
    print(f"Color Difference: {color_diff}")
    print(f"Histogram Distance: {hist_diff}")
    
    if color_diff < color_threshold and hist_diff < hist_threshold:
        print("Clothing matches!")
        sys.exit(0)  # Success
    else:
        print("Clothing not matches!")

        sys.exit(1)  # Failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match clothing color in images.")
    parser.add_argument("cloth_image", type=str, help="Path to the cloth image")
    parser.add_argument("person_image", type=str, help="Path to the person image")
    parser.add_argument("mask_image", type=str, help="Path to the mask image")
    args = parser.parse_args()
    
    match_clothing(args.cloth_image, args.person_image, args.mask_image)