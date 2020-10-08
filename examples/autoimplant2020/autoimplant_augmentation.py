from nipype.interfaces.ants import RegistrationSynQuick, ApplyTransforms
import nibabel as nib
from scipy import ndimage
import numpy as np
import os
import shutil
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case1", required=True)
    parser.add_argument("--case2", required=True)
    parser.add_argument("--n_threads", type=int, default=1)
    parser.add_argument("--directory", default="/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set")
    parser.add_argument("--output_directory",
                        default="/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set/registrations")
    return parser.parse_args()


def connected_v_not_connected(binary, minimum=1000):
    label_map, n_labels = ndimage.measurements.label(binary)
    labels = np.arange(1, n_labels)
    counts = ndimage.labeled_comprehension(label_map > 0, label_map, labels, np.sum, float, 0)
    foreground = label_map > 0
    keeper_labels = labels[counts > minimum]
    connected_mask = np.in1d(label_map, keeper_labels).reshape(binary.shape)
    not_connected_mask = np.logical_and(connected_mask == False, foreground)
    return connected_mask, not_connected_mask


def register_skull_to_skull(skull_filename1, skull_filename2, prefix, num_threads=1, debug=False):
    cmd = RegistrationSynQuick(fixed_image=skull_filename2, moving_image=skull_filename1,
                               output_prefix=prefix, num_threads=num_threads)
    print(cmd.cmdline)
    cmd.run()
    if not debug:
        for fn in glob.glob(os.path.abspath(os.path.join(".", prefix + "*Warped.nii.gz"))):
            print("Removing:", fn)
            os.remove(fn)


def apply_transforms(input_filename, reference_filename, transforms, output_filename,
                     interpolation="NearestNeighbor", args="-u uchar", num_threads=1,
                     invert_transform_flags=None):
    if invert_transform_flags is None:
        invert_transform_flags = [False for t in transforms]
    cmd = ApplyTransforms(transforms=transforms, input_image=input_filename, output_image=output_filename,
                          reference_image=reference_filename, interpolation=interpolation,
                          args=args, num_threads=num_threads,
                          invert_transform_flags=invert_transform_flags)
    print(cmd.cmdline)
    cmd.run()
    return cmd.output_spec().output_image


def apply_implant_to_skull(skull_filename, implant_filename):
    image = nib.load(skull_filename)
    implant_image = nib.load(implant_filename)
    data = np.copy(image.get_fdata())
    binary = data > 0

    implant_data = implant_image.get_fdata()
    implant_binary = implant_data > 0.5

    defective_binary = np.copy(binary)
    defective_binary[implant_binary] = 0

    opened_defective_binary = ndimage.binary_opening(defective_binary, iterations=1)

    implant_binary[np.logical_and(opened_defective_binary == False, defective_binary)] = 1

    connected, not_connected = connected_v_not_connected(opened_defective_binary, 1000)

    # add the not connected components to the implant mask
    implant_binary[not_connected] = 1

    connected_implant, trash = connected_v_not_connected(implant_binary, 1000)

    # mask the implant with the original skull image
    final_implant = np.logical_and(connected_implant, binary)

    # mask the skull with the implant
    final_defective = np.logical_and(binary, final_implant == False)

    final_defective_image = image.__class__(dataobj=np.asarray(final_defective, dtype='B'), affine=image.affine)
    final_implant_image = image.__class__(dataobj=np.asarray(final_implant, dtype='B'), affine=image.affine)
    return final_defective_image, final_implant_image


def check_if_transforms_exist(prefix, output_directory):
    return all([os.path.exists(fn) for fn in get_prefix_transforms(prefix, output_directory)])


def get_prefix_transforms(prefix, output_directory):
    affine_filename = os.path.join(output_directory, prefix + "0GenericAffine.mat")
    warp_filename = os.path.join(output_directory, prefix + "1Warp.nii.gz")
    inverse_warp_filename = os.path.join(output_directory, prefix + "1InverseWarp.nii.gz")
    return affine_filename, warp_filename, inverse_warp_filename


def get_filename(case, directory, sub_directory):
    return os.path.join(directory, sub_directory, case + ".nii.gz")


def get_skull(case, directory):
    return get_filename(case, directory, "complete_skull")


def get_implant(case, directory):
    return get_filename(case, directory, "implant")


def get_defective(case, directory):
    return get_filename(case, directory, "defective_skull")


def augment_image(case1, case2, directory, output_directory, transforms, name, invert_transform_flags=None,
                  num_threads=1):
    output_filename = os.path.join(output_directory, "augmented_" + name, 
                                   "sub-{}_space-{}_{}.nii.gz".format(case1, case2, name))
    if not os.path.exists(output_filename):
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename))
        apply_transforms(get_filename(case1, directory, name), get_skull(case2, directory=directory),
                         output_filename=output_filename,
                         transforms=transforms, invert_transform_flags=invert_transform_flags, num_threads=num_threads)
        

def augment_defective_skull(*args, **kwargs):
    return augment_image(*args, name="defective_skull", **kwargs)


def augment_implant(*args, **kwargs):
    return augment_image(*args, name="implant", **kwargs)


def copy_image(case, directory, output_directory, name):
    output_filename = os.path.join(output_directory, "augmented_" + name, 
                                   "sub-{}_space-{}_{}.nii.gz".format(case, case, name))
    if not os.path.exists(output_filename):
        input_filename = get_filename(case, directory, name)
        print("Copying:", input_filename, "-->", output_filename)
        shutil.copy(input_filename, output_filename)


def augment_auto_implant_cases(case1, case2, directory, output_directory, n_threads=1):
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # case 1 to case 2
    prefix1 = "_to_".join((case1, case2))
    prefix2 = "_to_".join((case2, case1))
    p1_exists = check_if_transforms_exist(prefix1, output_directory)
    p2_exists = check_if_transforms_exist(prefix2, output_directory)
    if not p1_exists and not p2_exists:
        # run registration with prefix1
        prefix = prefix1
        register_skull_to_skull(get_skull(case1, directory), get_skull(case2, directory), prefix,
                                num_threads=n_threads)
    elif p2_exists:
        # get transforms for prefix 2
        placeholder = case1
        case1 = case2
        case2 = placeholder
        prefix = prefix2
    else:
        # get transforms for prefix 1
        prefix = prefix1
    transforms = get_prefix_transforms(prefix, output_directory)
    # apply transforms
    for name in ("implant", "defective_skull", "complete_skull"):
        # augment defective skull
        augment_image(case1, case2, directory, output_directory, [transforms[1], transforms[0]],
                      num_threads=n_threads, name=name)
        augment_image(case2, case1, directory, output_directory, [transforms[0], transforms[2]],
                      invert_transform_flags=[True, False], num_threads=n_threads, name=name)
    # copy over non-augmented filenames
    for case in (case1, case2):
        for name in ("implant", "defective_skull", "complete_skull"):
            copy_image(case, directory, output_directory, name)


def main():
    augment_auto_implant_cases(**vars(parse_args()))


if __name__ == "__main__":
    main()
