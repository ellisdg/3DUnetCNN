"""
This script registers each of the atlas cases to the template, and then transforms each case by combining transforms.
"""

import glob
from nipype import Node, Workflow, IdentityInterface, MapNode, Function
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.ants.registration import RegistrationSynQuick, RegistrationSynQuickInputSpec, traits
import os


#TODO: add flair, t2, and t1c images


class _RegistrationSynQuickInputSpec(RegistrationSynQuickInputSpec):
    transform_type = traits.Enum(
        "s",
        "t",
        "r",
        "a",
        "sr",
        "so",
        "b",
        "br",
        argstr="-t %s",
        desc="""\
    Transform type

      * t:  translation
      * r:  rigid
      * a:  rigid + affine
      * s:  rigid + affine + deformable syn (default)
      * sr: rigid + deformable syn
      * b:  rigid + affine + deformable b-spline syn
      * br: rigid + deformable b-spline syn

    """,
        usedefault=True,
    )


class _RegistrationSynQuick(RegistrationSynQuick):
    input_spec = _RegistrationSynQuickInputSpec


def mix_n_match(t1_files, mask_files, forward_transforms, inverse_transforms):
    """
    Mix and match the transforms to form every combination of transformation possible.
    """
    t1_out = list()
    mask_out = list()
    transforms_out = list()
    reference_out = list()
    for t1, mask, forward in zip(t1_files, mask_files, forward_transforms):
        for reference, inverse in zip(t1_files, inverse_transforms):
            if reference != t1:
                transforms_out.append([inverse, forward])
                t1_out.append(t1)
                mask_out.append(mask)
                reference_out.append(reference)
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
        os.path.abspath("./RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/BraTS2021_*/BraTS2021_*_t1.nii.gz'"))
    mask_fns = list()
    reg_mask_fns = list()
    reg_mask_args = list()
    for t1_fn in t1_fns:
        mask_fn = t1_fn.replace("t1.", "seg.")
        mask_fns.append(mask_fn)
    #        reg_mask_arg = "-x Null,{}".format(reg_mask_fn)
    #        reg_mask_args.append(reg_mask_arg)

    reg_masker = MapNode(Function(function=create_reg_mask,
                                  input_names=["seg_file"],
                                  output_names=["reg_mask", "reg_mask_arg"]),
                         name="CreateRegMasks",
                         iterfield=["seg_file"])

    input_node = Node(IdentityInterface(["target", "t1s", "masks"]), name="inputnode")
    input_node.inputs.t1s = t1_fns
    input_node.inputs.masks = mask_fns
    input_node.inputs.target = "./SRI-24-Brain.nii.gz"

    wf.connect(input_node, "masks", reg_masker, "seg_file")

    reg_node = MapNode(_RegistrationSynQuick(transform_type="so"),
                       name="Registration", iterfield=["args", "moving_image"])
    wf.connect(input_node, "target", reg_node, "fixed_image")
    wf.connect(input_node, "t1s", reg_node, "moving_image")
    wf.connect(reg_masker, "reg_mask_arg", reg_node, "args")

    mixer = Node(Function(function=mix_n_match, outputs=["t1_out", "mask_out", "transforms_out", "reference_out"]),
                 name="MixNMatch")
    wf.connect(reg_node, "forward_warp_field", mixer, "forward_transforms")
    wf.connect(reg_node, "inverse_warp_field", mixer, "inverse_transforms")
    wf.connect(input_node, "t1s", mixer, "t1_files")
    wf.connect(input_node, "masks", mixer, "mask_files")

    transform_t1s = MapNode(ApplyTransforms(), name="TransformT1s",
                            iterfield=["reference_image", "input_image", "transforms"])
    wf.connect(mixer, "t1_out", transform_t1s, "input_image")
    wf.connect(mixer, "reference_out", transform_t1s, "reference_image")
    wf.connect(mixer, "transforms_out", transform_t1s, "transforms")

    transform_masks = MapNode(ApplyTransforms(interpolation="NearestNeighbor"), name="TransformMasks",
                              iterfield=["reference_image", "input_image", "transforms"])
    wf.connect(mixer, "mask_out", transform_masks, "input_image")
    wf.connect(mixer, "reference_out", transform_masks, "reference_image")
    wf.connect(mixer, "transforms_out", transform_masks, "transforms")

    syncer = Node(Function(function=sync_outputs), name="SyncOutputs")
    wf.connect(transform_t1s, "output_image", syncer, "warped_t1s")
    wf.connect(transform_masks, "output_image", syncer, "warped_masks")
    wf.connect(mixer, "mask_out", syncer, "mask_files")
    wf.connect(mixer, "t1_out", syncer, "t1_files")
    wf.connect(mixer, "reference_out", syncer, "reference_files")

    wf.run(plugin="MultiProc", plugin_args={"n_procs": 40})


if __name__ == "__main__":
    main()
