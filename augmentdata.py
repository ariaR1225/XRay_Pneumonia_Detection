import os
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_normal_class(base_train_dir, num_augmented=2300):

    normal_dir = os.path.join(base_train_dir, "NORMAL")
    augment_dir = os.path.join(base_train_dir, "NORMAL_augmented")
    os.makedirs(augment_dir, exist_ok=True)

    # Initialize the ImageDataGenerator with desired augmentations
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    batch_size = 32
    target_size = (255, 255)

    # Create an augmented image generator for the NORMAL class
    augment_generator = datagen.flow_from_directory(
        directory=base_train_dir,
        classes=['NORMAL'],
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        save_to_dir=augment_dir,
        save_prefix='aug_normal',
        save_format='jpeg'
    )

    # Generate the required number of augmented images
    batches = num_augmented // batch_size + 1
    for _ in tqdm(range(batches), desc="Generating augmented images"):
        next(augment_generator)

if __name__ == "__main__":
    
    base_train_dir = "chest_xray/preprocessed/train"
    augment_normal_class(base_train_dir, num_augmented=2300)