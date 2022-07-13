"""
This script registers each of the atlas cases to the template, and then transforms each case by combining transforms.
"""

import glob
from nipype import Node, Workflow, IdentityInterface, MapNode
from nipype.interfaces.ants import RegistrationSynQuick


def main():
    wf = Workflow("RegistrationWF")

    t1_fns = glob.glob(
        "/work/aizenberg/dgellis/MICCAI/2022/isles/isles_2022/data/train/derivatives/ATLAS/sub-*/ses-*/*/sub-*T1w.nii.gz")
    mask_fns = list()
    reg_mask_fns = list()
    reg_mask_args = list()
    for t1_fn in t1_fns:
        mask_fn = t1_fn.replace("T1w.", "label-L_desc-T1lesion_mask.")
        mask_fns.append(mask_fn)
        reg_mask_fn = t1_fn.replace("T1w.", "label-regmask_mask.")
        reg_mask_fns.append(reg_mask_fn)
        reg_mask_arg = "-x Null,{}".format(reg_mask_fn)
        reg_mask_args.append(reg_mask_arg)

    input_node = Node(IdentityInterface(["target", "t1s", "masks", "reg_masks", "reg_mask_args"]), name="inputnode")
    input_node.inputs.t1s = t1_fns
    input_node.inputs.masks = mask_fns
    input_node.inputs.reg_masks = reg_mask_fns
    input_node.inputs.reg_mask_args = reg_mask_args
    input_node.inputs.target = '/lustre/work/aizenberg/dgellis/MICCAI/2022/isles/isles_2022/tpl-MNI152NLin2009aSym_res-1_T1w.nii.gz'

    reg_node = MapNode(RegistrationSynQuick(transform_type="so"),
                       name="Registration", iterfield=["args", "moving_image"])
    wf.connect(input_node, "target", reg_node, "fixed_image")
    wf.connect(input_node, "t1s", reg_node, "moving_image")
    wf.connect(input_node, "reg_mask_args", reg_node, "args")

    wf.run(plugin="MultiProc", plugin_args={"n_procs": 40})


if __name__ == "__main__":
    main()
