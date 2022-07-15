"""
This script registers each of the atlas cases to the template, and then transforms each case by combining transforms.
"""

import glob
from nipype import Node, Workflow, IdentityInterface, MapNode, Function
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.ants.registration import RegistrationSynQuick, RegistrationSynQuickInputSpec, traits, File
import os


class _RegistrationSynQuickInputSpec(RegistrationSynQuickInputSpec):
    moving_mask = File(argstr="-x Null,%s")


class _RegistrationSynQuick(RegistrationSynQuick):
    input_spec = _RegistrationSynQuickInputSpec


def mix_n_match(t1_file, mask_file, t1c_file, flair_file, t2_file, forward_transform, forward_matrix,
                inverse_transforms, forward_matrices, reference_files, n=10,
                output_dir="/work/aizenberg/dgellis/MICCAI/2022/isles/isles_2022/augmented_data"):
    """
    Mix and match the transforms to form every combination of transformation possible.
    """
    import os
    from nipype.interfaces.ants import ApplyTransforms
    import random
    os.makedirs(output_dir, exist_ok=True)
    t1_out = list()
    mask_out = list()
    transforms_out = list()
    reference_out = list()
    output_images = list()
    _sub = t1_file.split("/")[-2]
    idx = list(range(len(reference_files)))
    random.shuffle(idx)
    idx = idx[:n]  # just do n transformations
    for i in idx:
        reference = reference_files[i]
        inverse = inverse_transforms[i]
        ref_forward_matrix = forward_matrices[i]
        ref_sub = reference.split("/")[-2]
        _output_dir = os.path.join(output_dir, "_".join((_sub, ref_sub)))
        if reference != t1_file:
            os.makedirs(_output_dir, exist_ok=True)
            for input_image in (t1_file, t1c_file, flair_file, t2_file, mask_file):
                output_image = os.path.join(_output_dir, os.path.basename(input_image))
                cmd = ApplyTransforms(transforms=[ref_forward_matrix, inverse, forward_transform, forward_matrix],
                                      invert_transform_flags=[True, False, False, False],
                                      reference_image=reference,
                                      input_image=input_image,
                                      output_image=output_image)
                if input_image == mask_file:
                    cmd.inputs.interpolation = "NearestNeighbor"
                cmd.run()
                output_images.append(output_image)

    return t1_out, mask_out, transforms_out, reference_out


def sync_outputs(t1_files, mask_files, reference_files, warped_t1s, warped_masks, output_dir="./augmented_data"):
    import shutil
    import os
    os.makedirs(output_dir, exist_ok=True)
    for t1, mask, ref, warped_t1, warped_mask in zip(t1_files, mask_files, reference_files, warped_t1s, warped_masks):
        t1_sub = os.path.basename(t1).split("_")[1]
        ref_sub = os.path.basename(ref).split("_")[1]
        _output_dir = os.path.join(output_dir, "_".join((t1_sub, ref_sub)))

        # copy t1
        out_t1 = os.path.join(_output_dir, os.path.basename(t1))
        if os.path.exists(out_t1):
            os.remove(out_t1)
        shutil.copy(warped_t1, out_t1)

        # copy mask
        out_mask = os.path.join(_output_dir, os.path.basename(mask))
        if os.path.exists(out_mask):
            os.remove(out_mask)
        shutil.copy(warped_mask, out_mask)


def create_reg_mask(seg_file):
    import nibabel as nib
    import numpy as np
    import os
    image = nib.load(seg_file)
    mask = np.asarray(np.asarray(image.dataobj) == 0, image.get_data_dtype())
    new_image = image.__class__(dataobj=mask, affine=image.affine)
    out_mask = os.path.abspath(os.path.basename(seg_file).replace("seg", "regmask"))
    new_image.to_filename(out_mask)
    reg_mask_arg = "-x Null,{}".format(out_mask)
    return out_mask, reg_mask_arg


def main():
    wf = Workflow("RegistrationWF")
    wf.base_dir = "./"

    t1_fns = glob.glob(
        os.path.abspath("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_*/BraTS2021_*_t1.nii.gz"))
    print("n:", len(t1_fns))
    t1c_fns = list()
    t2_fns = list()
    flair_fns = list()
    mask_fns = list()
    for t1_fn in t1_fns:
        t1c_fns.append(t1_fn.replace("t1.", "t1ce."))
        t2_fns.append(t1_fn.replace("t1.", "t2."))
        flair_fns.append(t1_fn.replace("t1.", "flair."))
        mask_fn = t1_fn.replace("t1.", "seg.")
        mask_fns.append(mask_fn)

    reg_masker = MapNode(Function(function=create_reg_mask,
                                  input_names=["seg_file"],
                                  output_names=["reg_mask", "reg_mask_arg"]),
                         name="CreateRegMasks",
                         iterfield=["seg_file"])

    input_node = Node(IdentityInterface(["target", "t1s", "t1cs", "t2s", "flairs", "masks"]), name="inputnode")
    input_node.inputs.t1s = t1_fns
    input_node.inputs.t1cs = t1c_fns
    input_node.inputs.t2s = t2_fns
    input_node.inputs.flairs = flair_fns
    input_node.inputs.masks = mask_fns
    input_node.inputs.target = os.path.abspath("./SRI-24-Brain.nii.gz")

    wf.connect(input_node, "masks", reg_masker, "seg_file")

    reg_node = MapNode(_RegistrationSynQuick(transform_type="s"),
                       name="Registration", iterfield=["moving_mask", "moving_image"])
    wf.connect(input_node, "target", reg_node, "fixed_image")
    wf.connect(input_node, "t1s", reg_node, "moving_image")
    wf.connect(reg_masker, "reg_mask", reg_node, "moving_mask")

    mixer = MapNode(Function(function=mix_n_match,
                             output_names=["t1_out", "mask_out", "transforms_out", "reference_out"]),
                    name="MixNMatch", iterfield=["t1_file", "mask_file", "forward_transform", "forward_matrix"])
    wf.connect(reg_node, "forward_warp_field", mixer, "forward_transform")
    wf.connect(reg_node, "inverse_warp_field", mixer, "inverse_transforms")
    wf.connect(reg_node, "out_matrix", mixer, "forward_matrix")
    wf.connect(reg_node, "out_matrix", mixer, "forward_matrices")
    wf.connect(input_node, "t1s", mixer, "t1_file")
    wf.connect(input_node, "t1s", mixer, "reference_files")
    wf.connect(input_node, "t1cs", mixer, "t1c_file")
    wf.connect(input_node, "t2s", mixer, "t2_file")
    wf.connect(input_node, "flairs", mixer, "flair_file")
    wf.connect(input_node, "masks", mixer, "mask_file")

    wf.run(plugin="MultiProc", plugin_args={"n_procs": 40})


if __name__ == "__main__":
    main()
