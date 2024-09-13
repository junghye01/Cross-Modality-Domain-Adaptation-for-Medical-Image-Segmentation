import numpy as np


# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    print(f'data shape!!!!!!:{data.shape}') # 4면 그대로 3이면 인덱스 수정.. 
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask) # segmentation mask 
    return nonzero_mask


def crop_to_nonzero(data:np.ndarray, seg, crop_size=(256,256),nonzero_label=-1):
    # c,x,y,z
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    
    nonzero_mask = create_nonzero_mask(seg)
    bbox = get_bbox_from_mask(nonzero_mask)

    zmin,zmax=bbox[0]
    ymin,ymax=bbox[2]
    xmin,xmax=bbox[1]

    x_center=(xmin+xmax)//2
    y_center=(ymin+ymax)//2

    # half size
    x_size_half=crop_size[0] // 2
    y_size_half=crop_size[1] // 2

    # start and end points of x,y, directions
    x_start=max(0, x_center-x_size_half)
    x_end=min(data.shape[2], x_center+x_size_half)

    y_start=max(0,y_center-y_size_half)
    y_end=min(data.shape[3],y_center+y_size_half)

    # apply slicer
    slicer=[slice(zmin,zmax),slice(x_start,x_end),slice(y_start,y_end)]
    data = data[tuple([slice(None)] + slicer)]

    if seg is not None:
        seg = seg[tuple([slice(None)] + slicer)]
    
    nonzero_mask = nonzero_mask[tuple(slicer)]
    
    if seg is not None:
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label #
    else:
        nonzero_mask = nonzero_mask.astype(np.int8)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask

    print(f'{y_start},{y_end},{x_start},{x_end},{x_center},{y_center}')
    return data, seg, bbox


