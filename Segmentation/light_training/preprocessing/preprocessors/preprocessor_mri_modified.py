#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Union, Tuple
import glob
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from light_training.preprocessing.cropping.cropping import crop_to_nonzero
# from .default_resampling import resample_data_or_seg_to_spacing, resample_img
from light_training.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from tqdm import tqdm
from light_training.preprocessing.normalization.default_normalization_schemes import CTNormalization, ZScoreNormalization
import SimpleITK as sitk 
from tqdm import tqdm 
from copy import deepcopy
import json 
import os
from .default_preprocessor_modified import DefaultPreprocessor

# mri 
from scipy.ndimage import zoom

class MultiModalityPreprocessor(DefaultPreprocessor):
    def __init__(self, 
                 base_dir,
                 image_dir,
                 
                 output_dir,
                 seg_dir=None,
                 data_filenames_pattern="*_0000.nii.gz",
                 seg_filename_pattern="*_MASK.nii.gz"):
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.output_dir = output_dir
        self.data_filenames_pattern = data_filenames_pattern
        self.seg_filename_pattern = seg_filename_pattern

    def get_iterable_list(self):
        # raw_data/imagesTr 안의 모든 케이스 파일명을 가져옵니다.
        all_cases = glob.glob(os.path.join(self.base_dir, self.image_dir, self.data_filenames_pattern))
        return all_cases

    def _normalize(self, data: np.ndarray, seg: np.ndarray,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            normalizer_class = ZScoreNormalization
            normalizer = normalizer_class(use_mask_for_norm=False,
                                          intensityproperties=foreground_intensity_properties_per_channel)
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def read_data(self, file_path):
        # 파일명에서 case_name 추출 (예: 10_SJB_Chest__1.5_B50f_AQ_0000.nii.gz -> 10_SJB_Chest__1.5_B50f_AQ)
        case_name = os.path.basename(file_path)[:-12]
        
        # 이미지 데이터 읽기
        d = sitk.ReadImage(file_path)
        spacing = d.GetSpacing()
        data = sitk.GetArrayFromImage(d).astype(np.float32)[None,]
        seg_arr = None

        # normalization
        data_min=np.min(data)
        data_max=np.max(data)
        data=(data-data_min)/(data_max-data_min)

        print(f'data 정규화 이후:{np.any(data==0)}')
        # 세그멘테이션 파일 경로 설정
        if self.seg_dir is not None:
            seg_file = os.path.join(self.base_dir, self.seg_dir, f"{case_name}_MASK.nii.gz")
            #print(f'seg_file:{seg_file}')
            
            if seg_file:
                seg = sitk.ReadImage(seg_file)
                seg_arr = sitk.GetArrayFromImage(seg).astype(np.float32)
                seg_arr = seg_arr[None]
                
                seg_arr = zoom(seg_arr, (1, 1, data.shape[2] / seg_arr.shape[2], data.shape[3] / seg_arr.shape[3]), order=0)

                print(f'data shape:{data.shape},seg_arr shape:{seg_arr.shape}') # c,x,y,z
                intensities_per_channel, intensity_statistics_per_channel = self.collect_foreground_intensities(seg_arr, data)
            else:
                print(f'no seg_file : {seg_file}')
                intensities_per_channel = []
                intensity_statistics_per_channel = []
        else:
            print(f'no seg_dir')
            intensities_per_channel = []
            intensity_statistics_per_channel = []


        properties = {"spacing": spacing, 
                      "raw_size": data.shape[1:], 
                      "name": case_name,
                      "intensities_per_channel": intensities_per_channel,
                      "intensity_statistics_per_channel": intensity_statistics_per_channel}

        # mri
        return data, seg_arr, properties 

    def run(self, 
            all_labels,
            output_spacing=None, 
            num_processes=8):
        self.out_spacing = output_spacing
        self.all_labels = all_labels
        self.foreground_intensity_properties_per_channel = {}

        all_iter = self.get_iterable_list()
        print(f'전체 파일 개수 : {len(all_iter)}')
        maybe_mkdir_p(self.output_dir)

        # test_run 
        for file_path in all_iter:
            #print(f'file_path:{file_path}')
            self.run_case_save(file_path)
          

        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            for file_path in all_iter:
                r.append(p.starmap_async(self.run_case_save,
                                         ((file_path,))))
            remaining = list(range(len(all_iter)))
            workers = [j for j in p._pool]
            with tqdm(desc=None, total=len(all_iter)) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    for _ in done:
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)
