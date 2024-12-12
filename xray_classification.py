import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import shutil
import tempfile

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available:")
        for gpu in gpus:
            print(f"  {gpu}")
        try:
            # Set memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU detected. Running on CPU")
    return gpus

class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, total_batches):
        super().__init__()
        self.epochs = epochs
        self.total_batches = total_batches
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f'\nEpoch {epoch+1}/{self.epochs}')
        self.progress_bar = tqdm(total=self.total_batches, 
                               desc='Training',
                               bar_format='{l_bar}{bar:30}{r_bar}')
        
    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)
        self.progress_bar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'accuracy': f"{logs['accuracy']:.4f}"
        })
        
    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()
        print(f"val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}")

class DataGeneratorWithProgress:
    def __init__(self, generator, total_samples, batch_size, desc):
        self.generator = generator
        self.total_batches = math.ceil(total_samples / batch_size)
        self.desc = desc
        
    def __iter__(self):
        with tqdm(total=self.total_batches, desc=self.desc) as pbar:
            for i, batch in enumerate(self.generator):
                if i >= self.total_batches:
                    break
                yield batch
                pbar.update(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Train X-ray classification model')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train')
    parser.add_argument('--data_dir', type=str, default='chest_xray',
                      help='Path to the data directory')
    return parser.parse_args()

def build_model(input_shape):
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the pre-trained layers
    base_model.trainable = False
    
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_data_generators(image_size, batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen

def prepare_datasets(data_dir, train_datagen, val_test_datagen, image_size, batch_size):
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'preprocessed/train'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'preprocessed/val'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'preprocessed/test'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall for positive class
    specificity = tn / (tn + fp)  # Recall for negative class
    
    print("\nDetailed Metrics from Confusion Matrix:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall for Pneumonia): {sensitivity:.4f}")
    print(f"Specificity (Recall for Normal): {specificity:.4f}")

def evaluate_model(model, test_generator):
    # Get predictions with progress bar
    test_generator.reset()
    total_samples = test_generator.n
    batch_size = test_generator.batch_size
    
    # Initialize arrays for predictions and true labels
    y_pred_probs = np.zeros(total_samples)
    y_true = test_generator.classes
    
    # Create progress bar for prediction
    test_steps = math.ceil(total_samples / batch_size)
    
    print("\nGenerating predictions:")
    for i, (x, _) in enumerate(tqdm(test_generator, total=test_steps, desc='Predicting')):
        if i >= test_steps:
            break
        batch_predictions = model.predict(x, verbose=0)
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        y_pred_probs[start_idx:end_idx] = batch_predictions.flatten()
    
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Plot confusion matrix
    class_names = ['Normal', 'Pneumonia']
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=class_names,
                              digits=4))
    
    return y_true, y_pred

def train_model():
    args = parse_args()
    
    # Check GPU availability
    gpus = check_gpu()
    
    print("Creating data generators...")
    train_datagen, val_test_datagen = create_data_generators(
        args.image_size, 
        args.batch_size
    )
    
    print("Preparing datasets...")
    train_generator, val_generator, test_generator = prepare_datasets(
        args.data_dir,
        train_datagen,
        val_test_datagen,
        args.image_size,
        args.batch_size
    )
    
    print("Building model...")
    model = build_model((args.image_size, args.image_size, 3))
    
    # Modified metrics to handle binary classification correctly
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')]
)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Calculate steps per epoch
    steps_per_epoch = math.ceil(train_generator.n / args.batch_size)
    validation_steps = math.ceil(val_generator.n / args.batch_size)
    
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
    # Train model with custom progress bar
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            TqdmCallback(args.epochs, steps_per_epoch),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'results/best_model.keras',
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.CSVLogger('results/training_history.csv')
        ],
        verbose=0
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    y_true, y_pred = evaluate_model(model, test_generator)
    
    return model, history

if __name__ == '__main__':
    model, history = train_model()