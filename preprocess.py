import cv2
import numpy as np
import pywt
from tqdm import tqdm
import os
from pathlib import Path

def wavelet_denoising(img):
    # Convert to float for wavelet transform
    img_float = img.astype(float)
    
    # Wavelet decomposition
    coeffs = pywt.wavedec2(img_float, 'db4', level=2)
    
    # Apply BayesShrink thresholding
    coeffs_thresholded = list(coeffs)
    for i in range(1, len(coeffs)):
        # For each detail coefficient level
        for j in range(3):
            # Estimate noise variance using median absolute deviation
            sigma = np.median(np.abs(coeffs[i][j])) / 0.6745
            
            # Estimate signal variance
            if sigma == 0:  # Avoid division by zero
                threshold = 0
            else:
                # Calculate variance of signal
                var = max(0, np.var(coeffs[i][j]) - sigma**2)
                # BayesShrink formula
                threshold = (sigma**2 / np.sqrt(var)) if var > 0 else sigma
            
            # Apply soft thresholding
            coeffs_thresholded[i] = tuple(
                pywt.threshold(c, threshold, mode='soft') 
                if k == j else c 
                for k, c in enumerate(coeffs[i])
            )
    
    # Inverse wavelet transform
    denoised = pywt.waverec2(coeffs_thresholded, 'db4')
    return denoised.astype(np.uint8)

def preprocess_xray(img_path, output_path):
    """
    Preprocess a single X-ray image and save it to the output path
    """
    # Read the image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    # Apply wavelet denoising
    denoised_wavelet = wavelet_denoising(img)
    
    # Apply bilateral filtering
    denoised_bilateral = cv2.bilateralFilter(denoised_wavelet, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised_bilateral)
    
    # Save the preprocessed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), enhanced)

def batch_preprocess(folder_name, input_dir, output_dir):
    """
    Preprocess all images in the training set while maintaining the directory structure
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all image files recursively
    image_files = list(input_path.rglob("*.jpeg")) + list(input_path.rglob("*.jpg"))
    
    print(f"Found {len(image_files)} images to process in {folder_name}")
    
    # Process each image with progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        # Create corresponding output path
        relative_path = img_path.relative_to(input_path)
        output_file = output_path / relative_path
        
        # Preprocess and save the image
        preprocess_xray(img_path, output_file)

if __name__ == "__main__":
    folder_names = ["train", "test", "val"]
    for n in folder_names:
        # Define input and output directories
        input_dir = f"chest_xray/{n}"
        output_dir = f"chest_xray/preprocessed/{n}"
        
        # Process all images
        batch_preprocess(n, input_dir, output_dir)