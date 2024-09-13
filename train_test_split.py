import argparse
import os
from tqdm import tqdm
import random
import shutil

def create_directories(paths):
    """Create directories for storing train and test images and labels."""
    for d, p in paths.items():
        os.makedirs(p, exist_ok=True)
        print(f'{d} dir created')

def copy_files(prefix_list, image_dir, label_dir, output_image_dir, output_label_dir, desc):
    """Copy image and label files from source to destination directories."""
    for prefix in tqdm(prefix_list, desc=f'Copying {desc}'):
        image_name = f'{prefix}_0000.nii.gz'
        mask_name = f'{prefix}_MASK.nii.gz'

        image_file = os.path.join(image_dir, image_name)
        mask_file = os.path.join(label_dir, mask_name)

        out_image_file = os.path.join(output_image_dir, image_name)
        out_mask_file = os.path.join(output_label_dir, mask_name)

        try:
            shutil.copy(image_file, out_image_file)
            shutil.copy(mask_file, out_mask_file)
        except FileNotFoundError as e:
            print(f"Error copying files: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

def train_test_split(image_dir, label_dir, output_dir):
    random.seed(1234)
    # Get list of image prefixes (removing suffix to generalize)
    image_prefix_list = sorted([file_name[:-12] for file_name in os.listdir(image_dir)])

    # Shuffle and split the data
    random.shuffle(image_prefix_list)
    split_index = int(len(image_prefix_list) * 0.9)
    train_prefix = image_prefix_list[:split_index]
    test_prefix = image_prefix_list[split_index:]

    # Define output directories
    paths = {
        'imagesTr': os.path.join(output_dir, 'imagesTr'),
        'LabelsTr': os.path.join(output_dir, 'LabelsTr'),
        'imagesTs': os.path.join(output_dir, 'imagesTs'),
        'LabelsTs': os.path.join(output_dir, 'LabelsTs')
    }

    # Create necessary directories
    create_directories(paths)

    # Copy files for train and test datasets
    copy_files(train_prefix, image_dir, label_dir, paths['imagesTr'], paths['LabelsTr'], 'train')
    copy_files(test_prefix, image_dir, label_dir, paths['imagesTs'], paths['LabelsTs'], 'test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train-test split script')
    parser.add_argument('--image_dir', type=str, default='./image_dir', help='Directory containing images')
    parser.add_argument('--label_dir', type=str, default='./label_dir', help='Directory containing labels')
    parser.add_argument('--output_dir', type=str, default='./output_dir', help='Output directory for split data')
    args = parser.parse_args()

    train_test_split(args.image_dir, args.label_dir, args.output_dir)




        
    