import os
from leap_binder import *
from os import environ
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
import onnxruntime

prediction_type1 = PredictionTypeHandler('opt_flow', ["x", "y"], channel_dim=-1)

@tensorleap_load_model([prediction_type1])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/raft_new.onnx'
    sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))
    return sess


@tensorleap_integration_test()
def check_custom_integration(idx, subset):  # This test requires the relevant secret to be loaded to the system environment AUTH_SECRET
    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")
    sess = load_model()
    # get inputs
    img_1 = input_image1(idx, subset)
    img_2 = input_image2(idx, subset)
    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    pred = sess.run(None, {input_name_1: img_1,
                                   input_name_2: img_2})[0]
    gt = gt_encoder(idx, subset)
    foreground_mask = fg_mask(idx, subset)

    # get all metadata
    idx = metadata_idx(idx, subset)
    filename = metadata_filename(idx, subset)
    name = dataset_name(idx, subset)
    focus_of_expansion = metadata_focus_of_expansion(idx, subset)
    metadata_all = metadata_dict(idx, subset)

    # run all visualizers and plot them. Visualizers run on results without the Batch dimension.
    img_vis = image_visualizer(img_1)
    visualize(img_vis)
    pred_flow_vis = flow_visualizer(pred)
    visualize(pred_flow_vis)
    gt_flow_vis = flow_visualizer(gt)
    visualize(gt_flow_vis)
    mask_vis = mask_visualizer(foreground_mask)
    visualize(mask_vis)
    # run losses and EPE
    loss = EPE(gt, pred)
    sample_fl_metric = fl_metric(gt, pred)
    sample_fl_fg_metric = fl_foreground(gt, pred, foreground_mask)
    sample_fl_bg_metric = fl_background(gt, pred, foreground_mask)
    print("Custom tests finished successfully")


if __name__ == "__main__":
    subsets = subset_images()
    check_custom_integration(0, subsets[0])
leap_binder.add_prediction('opt_flow', ['x', 'y'])
