import os
import shutil
import cv2
import numpy as np
import pandas as pd
from utils.file_utilities import get_latest_directory

def validate_and_preprocess(output_dir, **kwargs):
    """
    Validates and preprocesses color and depth images.

    :param kwargs:
        -   color_dir = path

        -   depth_dir = path

        -   scan_dir = path

        -   commands_path = path
    """

    color_image_dir = kwargs.get('color_dir', False)
    depth_image_dir = kwargs.get('depth_dir', False)
    scan_dir = kwargs.get('scan_dir', False)
    commands_path = kwargs.get('commands_path', False)

    os.makedirs(output_dir, exist_ok=True)

    if color_image_dir:
        processed_color_dir = os.path.join(output_dir, "color_images")
        os.makedirs(processed_color_dir, exist_ok=True)
        print(f"Processing color images from {color_image_dir}...")
        process_images(color_image_dir, processed_color_dir, expected_shape=(360, 640, 3), is_depth=False)
    else:
        processed_color_dir = False

    if depth_image_dir:
        processed_depth_dir = os.path.join(output_dir, "depth_images")
        os.makedirs(processed_depth_dir, exist_ok=True)
        print(f"Processing depth images from {depth_image_dir}...")
        process_images(depth_image_dir, processed_depth_dir, expected_shape=(360, 640, 1), is_depth=True)
    else:
        processed_depth_dir = False

    if scan_dir:
        processed_scan_dir = os.path.join(output_dir, 'scans')
        os.makedirs(processed_scan_dir, exist_ok=True) # TODO how will LiDAR data be implemented?
        print(f"Processing scans from {scan_dir}...")
        pass # TODO
    
    if commands_path:
        print(f"Processing commands.csv from {commands_path}")
        process_commands(commands_path, output_dir, 
                         color_dir = processed_color_dir,
                         depth_dir = processed_depth_dir)
        
    print("Preprocessing completed successfully.")


def process_commands(commands_path, output_dir, **kwargs):
    """
    Processes the commands.csv file to update paths for color and depth images.

    :param commands_path: Path to the commands.csv file
    :param output_dir: Output directory for preprocessed data
    :param kwargs:
        -   color_dir: Directory containing color images
        -   depth_dir: Directory containing depth images
    """
    color_dir = kwargs.get('color_dir', False)
    depth_dir = kwargs.get('depth_dir', False)

    # Load the CSV file into a DataFrame
    commands_df = pd.read_csv(commands_path)

    # Update the paths in the relevant columns
    if color_dir:
        print("Processing color image filenames...")
        if 'color_image_filename' in commands_df.columns:
            commands_df['color_image_filename'] = commands_df['color_image_filename'].apply(
                lambda x: os.path.join(color_dir, os.path.splitext(x)[0] + '.npy') if pd.notnull(x) else x
            )

    if depth_dir:
        print("Processing depth image filenames...")
        if 'depth_image_filename' in commands_df.columns:
            commands_df['depth_image_filename'] = commands_df['depth_image_filename'].apply(
                lambda x: os.path.join(depth_dir, os.path.splitext(x)[0] + '.npy') if pd.notnull(x) else x
            )

    # Save the updated DataFrame back to a CSV
    processed_commands_path = os.path.join(output_dir, "commands.csv")
    commands_df.to_csv(processed_commands_path, index=False)

    print(f"Commands file processed and saved to {processed_commands_path}.")

def process_images(input_dir, output_dir, expected_shape, is_depth):
    """
    Processes images by validating their format and preprocessing them.

    Args:
        input_dir (str): Path to the input images directory.
        output_dir (str): Path to save processed images.
        expected_shape (tuple): Expected shape of the images (H, W, C).
        is_depth (bool): Flag indicating whether the images are depth images.
    """
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Load the image
        if is_depth:
            image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)  # Depth: unchanged for raw values
        else:
            image = cv2.imread(input_path)  # Color: load as BGR

        if image is None:
            print(f"Warning: Could not load image {input_path}")
            continue

        # Validate shape
        if is_depth:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension for depth
        if image.shape != expected_shape:
            print(f"Invalid shape for {input_path}. Expected {expected_shape}, got {image.shape}.")
            continue

        # Normalize
        if is_depth:
            max_depth = np.max(image)
            if max_depth == 0:
                print(f"Invalid depth image at {input_path}, max depth is zero.")
                continue
            image = image / max_depth  # Normalize depth to [0, 1]
        else:
            image = image / 255.0  # Normalize color to [0, 1]

        # Save the processed image
        np.save(output_path.replace('.png', '.npy').replace('.jpg', '.npy'), image)
        print(f"Processed and saved: {output_path.replace('.png', '.npy').replace('.jpg', '.npy')}")


def main():

    # Set directories for preprocessing
    extracted_rosbag_dir = os.path.join('data', 'rosbags_extracted')

    latest_extracted_rosbag = get_latest_directory(extracted_rosbag_dir)

    if latest_extracted_rosbag is None:
        raise FileNotFoundError # TODO improve this to log message

    color_image_dir = os.path.join(latest_extracted_rosbag, 'color_images')
    depth_image_dir = os.path.join(latest_extracted_rosbag, 'depth_images')
    scan_dir = os.path.join(latest_extracted_rosbag, 'scans')
    commands_path = os.path.join(latest_extracted_rosbag, 'commands.csv')

    output_dir = os.path.join('data', 'processed_data', os.path.basename(latest_extracted_rosbag))

    validate_and_preprocess(output_dir, 
                            color_dir=color_image_dir,
                            depth_dir=depth_image_dir,
                            scan_dir = scan_dir,
                            commands_path = commands_path)


if __name__ == '__main__':
    main()
