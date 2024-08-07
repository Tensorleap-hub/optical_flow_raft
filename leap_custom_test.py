import os

import matplotlib.pyplot as plt
from leap_binder import *
import onnxruntime as rt
from os import environ
from os.path import exists
import urllib


def check_custom_integration():  # This test requires the relevant secret to be loaded to the system environment AUTH_SECRET
    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = ('model/raft_new.onnx')
    sess = rt.InferenceSession(os.path.join(dir_path, model_path))

    data_subsets: List[PreprocessResponse] = subset_images()  # preprocess and get data preprocess response list
    i, scene_flow = 0, data_subsets[0]
    x = subset_images()
    # get inputs
    img_1 = np.expand_dims(input_image1(i, scene_flow), axis=0)
    img_2 = np.expand_dims(input_image2(i, scene_flow), axis=0)
    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    label_name = sess.get_outputs()[-1].name
    pred = sess.run([label_name], {input_name_1: np.moveaxis(img_1.astype(np.float32), [1, 2, 3], [2, 3, 1]),
                                   input_name_2: np.moveaxis(img_2.astype(np.float32), [1, 2, 3], [2, 3, 1])})[
        0]
    pred = np.moveaxis(pred, [1, 2, 3], [3, 1, 2])
    # get gt and add batch so its shape would be [1,H,W,2]
    gt = gt_encoder(i, x[0])[None, ...]
    # tensorflow_gt = tf.convert_to_tensor(gt)
    # get foreground mask and add batch dimension
    foreground_mask = fg_mask(i, scene_flow)[None, ...]

    # get all metadata
    idx = metadata_idx(i, scene_flow)
    filename = metadata_filename(i, scene_flow)
    name = dataset_name(i, scene_flow)
    focus_of_expansion = metadata_focus_of_expansion(i, scene_flow)
    metadata_all = metadata_dict(i, scene_flow)

    # run all visualizers and plot them. Visualizers run on results without the Batch dimension.
    img_vis = image_visualizer(img_1[0, ...])
    plt.imshow(img_vis.data)
    pred_flow_vis = flow_visualizer(pred[0, ...])
    plt.imshow(pred_flow_vis.data)
    gt_flow_vis = flow_visualizer(gt[0, ...])
    plt.imshow(gt_flow_vis.data)
    mask_vis = mask_visualizer(foreground_mask[0, ...])
    plt.imshow(mask_vis.data)
    # run losses and EPE
    loss = EPE(gt, pred)
    sample_fl_metric = fl_metric(gt, pred)
    sample_fl_fg_metric = fl_foreground(gt, pred, foreground_mask)
    sample_fl_bg_metric = fl_background(gt, pred, foreground_mask)
    print("Custom tests finished successfully")


if __name__ == "__main__":
    check_custom_integration()
