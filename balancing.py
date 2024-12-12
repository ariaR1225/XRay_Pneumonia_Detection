import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_balanced_dataset(source_dir, dest_dir, train_size=0.8, val_size=0.1):
    """
    Create a balanced dataset with proper train/val/test split.
    
    Args:
        source_dir: Original data directory containing 'NORMAL' and 'PNEUMONIA' subdirectories
        dest_dir: Destination directory for the balanced dataset
        train_size: Proportion of data for training
        val_size: Proportion of data for validation (remaining goes to test)
    """
    
    # Create destination directories
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)
    
    # Get file lists for each class
    normal_files = os.listdir(os.path.join(source_dir, 'NORMAL'))
    pneumonia_files = os.listdir(os.path.join(source_dir, 'PNEUMONIA'))
    
    # Balance the classes by undersampling the majority class
    min_samples = min(len(normal_files), len(pneumonia_files))
    normal_files = np.random.choice(normal_files, min_samples, replace=False)
    pneumonia_files = np.random.choice(pneumonia_files, min_samples, replace=False)
    
    # Split the data for both classes
    for files, cls in [(normal_files, 'NORMAL'), (pneumonia_files, 'PNEUMONIA')]:
        # First split: separate training set
        train_files, temp_files = train_test_split(files, train_size=train_size, random_state=42)
        
        # Second split: separate validation and test from the remaining data
        val_files, test_files = train_test_split(
            temp_files, 
            train_size=val_size/(1-train_size),
            random_state=42
        )
        
        # Copy files to respective directories
        for file_list, split in [(train_files, 'train'), 
                               (val_files, 'val'), 
                               (test_files, 'test')]:
            for file in file_list:
                src = os.path.join(source_dir, cls, file)
                dst = os.path.join(dest_dir, split, cls, file)
                shutil.copy2(src, dst)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    for split in splits:
        print(f"\n{split.capitalize()} Set:")
        for cls in classes:
            count = len(os.listdir(os.path.join(dest_dir, split, cls)))
            print(f"{cls}: {count} images")

def create_augmented_data_generators(image_size, batch_size):
    """
    Create data generators with augmentation for training
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen

# Usage example
if __name__ == '__main__':
    # Set up directories
    source_dir = 'chest_xray/train'  # Original data directory
    balanced_dir = 'chest_xray/balanced'  # New balanced dataset directory
    
    # Create balanced dataset
    create_balanced_dataset(source_dir, balanced_dir)