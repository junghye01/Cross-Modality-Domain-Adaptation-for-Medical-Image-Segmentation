import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse


def read_npz_data(file):
    with np.load(file) as f:
        seg = f['seg']
    return seg

def compute_class_weights(labels, num_classes):
    class_counts = np.zeros(num_classes)  # Changed to use a fixed-size array
    total_pixels = 0

    # Count the number of pixels for each class
    for label in tqdm(labels,desc='Counting pixels for each class'):
        unique, counts = np.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            if u != -1 and u < num_classes:  # Exclude labels that are not in range
                class_counts[u] += c
        total_pixels += label.size

    # Compute class weights
    print(f'total # pixels : {total_pixels}')
    class_weights = total_pixels / ((num_classes - 1) * class_counts)  # Calculate class weights
    class_weights = np.where(class_counts > 0, class_weights, 0)  # Set weight to 0 for classes with 0 pixels

    # Normalize weights to sum to 1
    weight_sum = np.sum(class_weights)
    normalized_class_weights = class_weights / weight_sum

    return normalized_class_weights

def calculate_weights_for_train_dataset(data_dir, num_classes):
    # Collect all segmentation maps
    all_labels = []

    for file in os.listdir(data_dir):
        if file.endswith('.npz'):
            seg = read_npz_data(os.path.join(data_dir, file))
            all_labels.append(seg)

    # Compute class weights
    weights = compute_class_weights(all_labels, num_classes)
    return weights


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Computing class weights')
    parser.add_argument('--data_dir',type=str,default='../../0_lymph_node_segmentation/dataset/cross_moda/preprocessed_data/s3_Tr')
    parser.add_argument('--num_classes',type=int,default=3)
    args=parser.parse_args()

    weights=calculate_weights_for_train_dataset(args.data_dir,args.num_classes)
    print(f'Computed Normalized class weights : {weights}')