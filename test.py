import os

def count_files_in_directories(base_path):
    """
    Count files in NORMAL and PNEUMONIA subdirectories
    
    Args:
        base_path (str): Base path to the chest_xray/preprocessed/train directory
    
    Returns:
        dict: Dictionary containing file counts for each category
    """
    categories = ['NORMAL', 'PNEUMONIA']
    file_counts = {}
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        try:
            # Count only files, not subdirectories
            files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            file_counts[category] = len(files)
        except FileNotFoundError:
            print(f"Directory not found: {category_path}")
            file_counts[category] = 0
            
    return file_counts

# Path to the training directory
base_directory = "chest_xray/train"

# Get the file counts
counts = count_files_in_directories(base_directory)

# Print results
print("\nNumber of files in training directories:")
print("-" * 35)
for category, count in counts.items():
    print(f"{category}: {count} files")
print("-" * 35)
print(f"Total files: {sum(counts.values())}")