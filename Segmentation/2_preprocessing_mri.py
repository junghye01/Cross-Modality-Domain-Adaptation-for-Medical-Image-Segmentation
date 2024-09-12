
from light_training.preprocessing.preprocessors.preprocessor_mri_modified import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 
import os
import argparse




def process_train():
    #base_dir='/mnt/nas203/forGPU2/junghye/CrossModa/dataset/segmamba/raw_data2'
    base_dir='/mnt/nas203/forGPU2/junghye/CrossModa/dataset/segmamba/s3_rawdata'
    output_dir='/mnt/nas203/forGPU2/junghye/0_lymph_node_segmentation/dataset/cross_moda/preprocessed_data'
    image_dir='imagesTr'
    seg_dir='LabelsTr'
    #seg_dir=None
    output_dir=os.path.join(output_dir,'s3_Tr')
    data_filename=os.listdir(os.path.join(base_dir,image_dir))
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                   # seg_dir=seg_dir,
                                    output_dir=output_dir,
                                    seg_dir=seg_dir,
                                   )

    #out_spacing = [1.0,1.0,1.0]
    out_spacing=None
    
    
    preprocessor.run(
                     all_labels=[1,2,],
                    output_spacing=out_spacing,
    )

"""
def plan():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()
"""


if __name__ == "__main__":

    #plan()
    process_train()

