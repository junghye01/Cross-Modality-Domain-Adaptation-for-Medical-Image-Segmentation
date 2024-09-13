import argparse
import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np



def convert_nifti_to_dcm(nifti_files, input_dir,output_dir):
    # NIfTI 파일 로드
    os.makedirs(output_dir,exist_ok=True)
    print(f'file length : {len(nifti_files)}')
    
    for nifti_file in tqdm(nifti_files,desc=f'Saving to {output_dir}'):
        image = sitk.ReadImage(os.path.join(input_dir,nifti_file))
        prefix=nifti_file[:-7]
        
        
        # Z축을 따라 각 슬라이스 저장
        for i in range(image.GetDepth()):
            # 각 슬라이스를 2D 이미지로 추출
            slice_image = image[:, :, i]
            slice_image.SetDirection((1, 0, 0, 1))  # 2D 방향을 의미하는 직교 행렬

            slice_image = sitk.Cast(slice_image, sitk.sitkInt16)
            
            # 메타데이터 설정
            slice_image.SetMetaData("0008|0060", "MR")  # Modality 설정 (MR 예시)
            slice_image.SetMetaData("0008|103E", "Description")  # Series Description 설정
            
            # DICOM 파일명 설정
            dicom_filename = os.path.join(output_dir, f"{prefix}_slice_{i:03d}.dcm")
            
            # DICOM 파일 저장
            sitk.WriteImage(slice_image, dicom_filename)


def convert_dcm_to_nifti(input_dir, output_dir, file_prefixes, z_spacing=1.0):
    """
    Convert a series of DICOM images starting with a specific prefix to a single 3D NIfTI file using SimpleITK.

    Args:
    - input_dir (str): Directory where DICOM files are located.
    - output_dir (str): Directory to save the output NIfTI files.
    - file_prefixes (list): List of prefixes of the DICOM files to be converted.
    - z_spacing (float): The spacing between slices in the Z direction (in mm).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each prefix
    for prefix in tqdm(file_prefixes, desc='Convert DICOM Images to NIfTI'):
        # List of DICOM files starting with the given prefix
        prefix=prefix[:-14]
        dicom_files = sorted([f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith('.dcm')])

        if not dicom_files:
            print(f"No DICOM files found with prefix '{prefix}' in directory {input_dir}")
            continue

        # Read the DICOM files into SimpleITK images
        images = []
        for dicom_file in dicom_files:
            file_path = os.path.join(input_dir, dicom_file)
            image = sitk.ReadImage(file_path)
            images.append(image)

        # Check if there are images to process
        if len(images) == 0:
            print(f"No images read for prefix '{prefix}'. Skipping.")
            continue

        # Combine the 2D DICOM slices into a 3D image
        try:
            image_4d = sitk.JoinSeries(images)
            
            
            if image_4d.GetDimension() == 4:
                # Convert RGB (4D) to Grayscale (3D)
                image_array = sitk.GetArrayFromImage(image_4d)  # Convert to numpy array
                print(f'Original image array shape: {image_array.shape}')  # Output shape: (120, 256, 256, 3)
                
                # Convert RGB to Grayscale using luminosity method
                grayscale_array = np.dot(image_array[...,:3], [0.2989, 0.587, 0.114])  # Shape will become (120, 256, 256)
                print(f'Converted grayscale image array shape: {grayscale_array.shape}')
                
                # Create a 3D SimpleITK image from the grayscale array
                image_3d = sitk.GetImageFromArray(grayscale_array.astype(np.float32))  # Ensure float32 type

                
                # Set the metadata for the 3D image to preserve spatial information
                spacing_2d = images[0].GetSpacing()  # Get 2D spacing (x, y)
                spacing_3d = (spacing_2d[0], spacing_2d[1], z_spacing)  # Extend to 3D spacing (x, y, z)
                image_3d.SetSpacing(spacing_3d)

                # Correctly extend direction to 3D
                direction_2d = images[0].GetDirection()  # Get 2D direction (4 elements)
                # Ensure the second row has valid entries to avoid determinant 0
                direction_3d = (
                    direction_2d[0], direction_2d[1], 0,  # First row
                    direction_2d[2], direction_2d[3], 0,  # Second row
                    0, 0, 1  # Third row for Z direction
                )
                # Ensure the determinant is not zero
                if np.linalg.det(np.array(direction_3d).reshape(3, 3)) == 0:
                    direction_3d = (
                        1, 0, 0,  # First row
                        0, 1, 0,  # Second row
                        0, 0, 1  # Third row for Z direction
                    )
                    print("Adjusted the direction matrix to avoid determinant 0.")

                image_3d.SetDirection(direction_3d)

                # Set the origin to match the first slice
                image_3d.SetOrigin(images[0].GetOrigin())

                # Save the 3D image as a NIfTI file
                output_path = os.path.join(output_dir, f'{prefix}_0000.nii.gz')
                sitk.WriteImage(image_3d, output_path)
                print(f"Saved NIfTI file: {output_path}")

            else:
                print(f"Unexpected image dimension for prefix '{prefix}': {image_4d.GetDimension()}")

        except Exception as e:
            print(f"Failed to convert DICOM series with prefix '{prefix}': {e}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='modality conversion')
    parser.add_argument('--input_dir',type=str,default='../query-selected-attention/results/qsattn_dicom/total_latest/images/fake_B')
    parser.add_argument('--output_dir',type=str,default='../dataset/segmamba/s2_rawdata/FakeB')
    parser.add_argument('--mode',type=str,default='n2d')
    args=parser.parse_args()

    if args.mode=='n2d':
        nifti_files=os.listdir(args.input_dir)
        convert_nifti_to_dcm(nifti_files,args.input_dir,args.output_dir)

    elif args.mode =='d2n':
        file_prefixes=set(os.listdir(args.input_dir))
        convert_dcm_to_nifti(args.input_dir,args.output_dir,file_prefixes)


    else:
        raise Exception('wrong mode name')