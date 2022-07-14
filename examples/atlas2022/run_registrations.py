"""
This script registers each of the atlas cases to the template, and then transforms each case by combining transforms.
"""

import glob
from nipype import Node, Workflow, IdentityInterface, MapNode, Function
from nipype.interfaces.ants import ApplyTransforms
from nipype.interfaces.ants.registration import RegistrationSynQuick, RegistrationSynQuickInputSpec, traits


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


def mix_n_match(t1_file, mask_file, forward_transform, inverse_transforms, reference_files,
                output_dir="/work/aizenberg/dgellis/MICCAI/2022/isles/isles_2022/augmented_data"):
    """
    Mix and match the transforms to form every combination of transformation possible.
    """
    import os
    from nipype.interfaces.ants import ApplyTransforms
    os.makedirs(output_dir, exist_ok=True)
    t1_out = list()
    mask_out = list()
    transforms_out = list()
    reference_out = list()
    output_images = list()
    _sub = t1_file.split("/")[-4]
    for reference, inverse in zip(reference_files, inverse_transforms):
        ref_sub = reference.split("/")[-4]
        _output_dir = os.path.join(output_dir, "_".join((_sub, ref_sub)))
        if reference != t1_file:
            os.makedirs(_output_dir, exist_ok=True)
            for input_image in (t1_file, mask_file):
                output_image = os.path.join(_output_dir, os.path.basename(input_image))
                cmd = ApplyTransforms(transforms=[inverse, forward_transform],
                                      reference_image=reference,
                                      input_image=t1_file,
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
        t1_sub = t1.split("/")[-4]
        ref_sub = ref.split("/")[-4]
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


def main():
    wf = Workflow("RegistrationWF")
    wf.base_dir = "./"

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

    reg_node = MapNode(_RegistrationSynQuick(transform_type="so"),
                       name="Registration", iterfield=["args", "moving_image"])
    wf.connect(input_node, "target", reg_node, "fixed_image")
    wf.connect(input_node, "t1s", reg_node, "moving_image")
    wf.connect(input_node, "reg_mask_args", reg_node, "args")

    mixer = MapNode(Function(function=mix_n_match,
                             output_names=["t1_out", "mask_out", "transforms_out", "reference_out"]),
                    name="MixNMatch", iterfield=["t1_file", "mask_file", "forward_transform"])
    wf.connect(reg_node, "forward_warp_field", mixer, "forward_transform")
    wf.connect(reg_node, "inverse_warp_field", mixer, "inverse_transforms")
    wf.connect(input_node, "t1s", mixer, "t1_file")
    wf.connect(input_node, "t1s", mixer, "reference_files")
    wf.connect(input_node, "masks", mixer, "mask_file")

    #transform_t1s = MapNode(ApplyTransforms(), name="TransformT1s",
    #                        iterfield=["reference_image", "input_image", "transforms"])
    #wf.connect(mixer, "t1_out", transform_t1s, "input_image")
    #wf.connect(mixer, "reference_out", transform_t1s, "reference_image")
    #wf.connect(mixer, "transforms_out", transform_t1s, "transforms")

    #transform_masks = MapNode(ApplyTransforms(interpolation="NearestNeighbor"), name="TransformMasks",
    #                          iterfield=["reference_image", "input_image", "transforms"])
    #wf.connect(mixer, "mask_out", transform_masks, "input_image")
    #wf.connect(mixer, "reference_out", transform_masks, "reference_image")
    #wf.connect(mixer, "transforms_out", transform_masks, "transforms")

    #syncer = Node(Function(function=sync_outputs), name="SyncOutputs")
    #wf.connect(transform_t1s, "output_image", syncer, "warped_t1s")
    #wf.connect(transform_masks, "output_image", syncer, "warped_masks")
    #wf.connect(mixer, "mask_out", syncer, "mask_files")
    #wf.connect(mixer, "t1_out", syncer, "t1_files")
    #wf.connect(mixer, "reference_out", syncer, "reference_files")

    wf.run(plugin="MultiProc", plugin_args={"n_procs": 40})


if __name__ == "__main__":
    main()
