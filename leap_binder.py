import cv2
from typing import List, Dict, Union
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import numpy.typing as npt
import json
from code_loader.contract.visualizer_classes import LeapImage
from code_loader.contract.enums import (
    LeapDataType
)
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess, tensorleap_input_encoder, \
    tensorleap_gt_encoder, tensorleap_metadata, tensorleap_custom_visualizer, tensorleap_custom_loss, \
    tensorleap_custom_metric
from numpy import ndarray

from optical_flow_raft.config import BUCKET_NAME, MAX_SCENE, MAX_STEREO, IMG_SIZE
from optical_flow_raft.data.preprocess import get_kitti_data
from optical_flow_raft.utils.flow_utils import decode_kitti_png, flow_to_image, EPE_mask, get_fl_map
from optical_flow_raft.utils.gcs_utils import download


# --------------------------------------------------inputs & GT---------------------------------------------------------
@tensorleap_preprocess()
def subset_images() -> List[PreprocessResponse]:
    scene_flow = get_kitti_data(bucket_name=BUCKET_NAME, data_subset="scene")
    stereo_flow = get_kitti_data(bucket_name=BUCKET_NAME, data_subset="stereo")
    scene_flow_poe_p = download("KITTI/data_scene_flow/estimated_poe/scene_flow_poe.json", bucket_name=BUCKET_NAME)
    with open(scene_flow_poe_p, 'r') as f:
        scene_flow_poe = json.load(f)
    stereo_flow_poe_p = download("KITTI/data_stereo_flow/estimated_poe/combined_poe.json", bucket_name=BUCKET_NAME)
    with open(stereo_flow_poe_p, 'r') as f:
        stereo_flow_poe = json.load(f)
    responses = [
        PreprocessResponse(length=min(len(scene_flow.train_IDs), MAX_SCENE),
                           data={"dataset_name": "scene_flow", 'paths': scene_flow.train_IDs,
                                 'poe': scene_flow_poe}),
        PreprocessResponse(length=min(len(stereo_flow.train_IDs), MAX_STEREO),
                           data={"dataset_name": "stereo_flow", 'paths': stereo_flow.train_IDs,
                                 'poe': stereo_flow_poe})
    ]
    return responses


def get_image(cloud_path: str) -> np.ndarray:
    fpath = download(str(cloud_path), bucket_name=BUCKET_NAME)
    flow_img = cv2.imread(fpath, -1)
    flow_img = cv2.resize(flow_img, IMG_SIZE[::-1])
    return flow_img


@tensorleap_input_encoder('image1', channel_dim=1)
def input_image1(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    path0 = data['paths'][idx][0]
    img0 = get_image(path0)
    return np.transpose(img0, [2,0,1]).astype(np.float32)

@tensorleap_input_encoder('image2', channel_dim=1)
def input_image2(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    path1 = data['paths'][idx][1]
    img1 = get_image(path1)
    return np.transpose(img1, [2,0,1]).astype(np.float32)

@tensorleap_gt_encoder('mask')
def gt_encoder(idx: int, data: PreprocessResponse) -> np.ndarray:
    data = data.data
    path = data['paths'][idx][2]
    img = get_image(path)
    img = decode_kitti_png(img)
    return img


# -------------------------------------------------------- metadata ----------------------------------------------------

def masked_of_percent(gt: np.ndarray) -> float:
    return 100 * gt[..., -1].sum() / (gt.shape[0] * gt.shape[1])


def average_of_magnitude(gt: np.ndarray) -> ndarray:
    return np.mean(np.sqrt(gt[gt[..., -1].astype(bool), 0] ** 2 + (gt[gt[..., -1].astype(bool), 1]) ** 2))


def mu_over_sigma_of(gt: np.ndarray) -> float:
    x_gt = gt[gt[..., -1] != 0, 0]
    y_gt = gt[gt[..., -1] != 0, 1]
    out_angles = np.divide(y_gt, x_gt, out=np.zeros_like(x_gt), where=x_gt != 0)
    std = out_angles.std()
    mean = out_angles.mean()
    if std > 0:
        return mean / std
    else:
        return mean

@tensorleap_metadata('metadata_dict')
def metadata_dict(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    gt = gt_encoder(idx, data)
    res = {
        "masked_of_percent": masked_of_percent(gt),
        "average_of_magnitude": np.float64(average_of_magnitude(gt)),
        "mu_over_sigma_of": np.float64(mu_over_sigma_of(gt))
    }
    return res


def poe_x(filename: str, data: PreprocessResponse) -> float:
    return data.data['poe'][filename][0]


def poe_y(filename: str, data: PreprocessResponse) -> float:
    return data.data['poe'][filename][1]

@tensorleap_metadata('metadata_focus_of_expansion')
def metadata_focus_of_expansion(idx: int, data: PreprocessResponse) -> Dict[str, Union[float, int, str]]:
    filename = metadata_filename(idx, data)
    res = {
        "focus_of_expansion_x": poe_x(filename, data),
        "focus_of_expansion_y": poe_y(filename, data)
    }
    return res

@tensorleap_metadata('filename')
def metadata_filename(idx: int, data: PreprocessResponse) -> str:
    data = data.data
    path = data['paths'][idx][0]
    fname = path.split('/')[-1]
    return fname

@tensorleap_metadata('dataset name')
def dataset_name(idx: int, subset: PreprocessResponse) -> str:
    return subset.data['dataset_name']

@tensorleap_metadata('idx')
def metadata_idx(idx: int, data: PreprocessResponse) -> int:
    return idx


# -------------------------------------------------------- visualizers -------------------------------------------------

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
    image = np.squeeze(image)
    return LeapImage(np.transpose(image, [1,2,0])[..., ::-1].astype(np.uint8))

@tensorleap_custom_visualizer('flow_visualizer', LeapDataType.Image)
def flow_visualizer(flow: npt.NDArray[np.float32]) -> LeapImage:
    flow = np.squeeze(flow)
    if flow.shape[0] == 2 or flow.shape[0] ==3: #channel first to last
        flow = np.transpose(flow, [1,2,0])
    img = flow_to_image(flow)
    return LeapImage(img)

@tensorleap_custom_visualizer('gt_visualizer', LeapDataType.Image)
def gt_visualizer(flow: npt.NDArray[np.float32]) -> LeapImage:
    img = flow_to_image(flow)
    img[(img == 255).all(axis=-1)] = 0
    return LeapImage(img)

@tensorleap_custom_visualizer('fg_visualizer', LeapDataType.Image)
def mask_visualizer(mask: npt.NDArray[np.uint8]) -> LeapImage:
    mask = np.squeeze(mask)
    return LeapImage((mask[..., None].repeat(3, axis=2) * 255).astype(np.uint8))


# -------------------------------------------------------- metrics  -------------------------------------------------

@tensorleap_custom_loss('EPE')
def EPE(gt_flow: np.ndarray, pred_flow: np.ndarray) -> tf.Tensor:
    gt_flow = tf.convert_to_tensor(gt_flow)
    pred_flow = np.transpose(pred_flow, [0, 2, 3, 1])
    pred_flow = tf.convert_to_tensor(pred_flow)
    pixel_err = EPE_mask(gt_flow, pred_flow)
    sample_err = tf.reduce_mean(pixel_err, axis=[1, 2])
    return sample_err.numpy()

@tensorleap_gt_encoder('fg_mask')
def fg_mask(idx: int, subset: PreprocessResponse) -> np.ndarray:
    if subset.data['dataset_name'] == 'scene_flow':
        filename = metadata_filename(idx, subset)
        local_file = download(f"KITTI/data_scene_flow/training/obj_map/{filename}", bucket_name=BUCKET_NAME)
        return np.array(Image.open(local_file).resize(IMG_SIZE[::-1], Image.NEAREST)).astype(np.float32)
    elif subset.data['dataset_name'] == 'stereo_flow':
        return np.ones_like(np.transpose(input_image1(idx, subset), [1,2,0])[..., 0]).astype(np.float32)

@tensorleap_custom_metric('FL-all')
def fl_metric(gt_flow: np.ndarray, pred_flow: np.ndarray) -> np.ndarray:
    gt_flow = tf.convert_to_tensor(gt_flow)
    pred_flow = np.transpose(pred_flow, [0, 2, 3, 1])# Channel First to Last
    pred_flow = tf.convert_to_tensor(pred_flow)
    fl_map = get_fl_map(gt_flow, pred_flow)
    outliers_num = tf.math.count_nonzero(fl_map, axis=[1, 2])
    return (outliers_num / (tf.maximum(tf.math.count_nonzero(gt_flow[..., -1], axis=[1, 2]), 1))).numpy()

@tensorleap_custom_metric('FL-fg')
def fl_foreground(gt_flow: np.ndarray, pred_flow: np.ndarray, foreground_map: np.ndarray) -> np.ndarray:
    gt_flow = tf.convert_to_tensor(gt_flow)
    pred_flow = np.transpose(pred_flow, [0, 2, 3, 1])# Channel First to Last
    pred_flow = tf.convert_to_tensor(pred_flow)
    foreground_map = tf.convert_to_tensor(foreground_map)
    fl_map = tf.cast(get_fl_map(gt_flow, pred_flow), float) * foreground_map
    outliers_num = tf.math.count_nonzero(fl_map, axis=[1, 2])
    combined_mask = gt_flow[..., -1] * foreground_map
    return (outliers_num / (tf.maximum(tf.math.count_nonzero(combined_mask, axis=[1, 2]), 1))).numpy()

@tensorleap_custom_metric('FL-bg')
def fl_background(gt_flow: np.ndarray, pred_flow: np.ndarray, foreground_map: np.ndarray) -> np.ndarray:
    gt_flow = tf.convert_to_tensor(gt_flow)
    pred_flow = np.transpose(pred_flow, [0, 2, 3, 1]) # Channel First to Last
    pred_flow = tf.convert_to_tensor(pred_flow)
    foreground_map = tf.convert_to_tensor(foreground_map)
    background_mask = 1 - foreground_map
    fl_map = tf.cast(get_fl_map(gt_flow, pred_flow), float) * background_mask
    outliers_num = tf.math.count_nonzero(fl_map, axis=[1, 2])
    combined_mask = gt_flow[..., -1] * background_mask
    return (outliers_num / (tf.maximum(tf.math.count_nonzero(combined_mask, axis=[1, 2]), 1))).numpy()


# -------------------------------------------------------- binding  -------------------------------------------------
# preprocess function
#leap_binder.set_preprocess(subset_images)

# set input and gt
#leap_binder.set_input(input_image1, 'image1')
#leap_binder.set_input(input_image2, 'image2')
#leap_binder.set_input(fg_mask, 'fg_mask')
#leap_binder.set_ground_truth(gt_encoder, 'mask')

# set prediction
#leap_binder.add_prediction()

# set meata_data
#leap_binder.set_metadata(metadata_idx, 'idx')
#leap_binder.set_metadata(metadata_filename, 'filename')
#leap_binder.set_metadata(dataset_name, 'dataset name')
#leap_binder.set_metadata(metadata_focus_of_expansion, 'metadata_focus_of_expansion')
#leap_binder.set_metadata(metadata_dict, 'metadata_dict')

# set visualizer
#leap_binder.set_visualizer(image_visualizer, 'image_visualizer', LeapDataType.Image)
#leap_binder.set_visualizer(flow_visualizer, 'flow_visualizer', LeapDataType.Image)
#leap_binder.set_visualizer(gt_visualizer, 'gt_visualizer', LeapDataType.Image)
#leap_binder.set_visualizer(mask_visualizer, 'fg_visualizer', LeapDataType.Image)

# set loss
#leap_binder.add_custom_loss(EPE, 'EPE')

# set custom metrics
#leap_binder.add_custom_metric(fl_metric, 'FL-all')
#leap_binder.add_custom_metric(fl_foreground, 'FL-fg')
#leap_binder.add_custom_metric(fl_background, 'FL-bg')

if __name__ == '__main__':
    leap_binder.check()
