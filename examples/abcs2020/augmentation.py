import glob
import os
import subprocess
from nipype.interfaces.ants import ApplyTransforms
import shutil
from functools import partial
from multiprocessing import Pool


def main():
    directory = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data"
    transforms_directory = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/augmentation"
    output_directory = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_augmented_training_data"
    num_threads = 16
    filenames = glob.glob(os.path.join(directory, "*.nii.gz"))
    subjects = get_subjects()
    if num_threads > 1:
        pool = Pool(num_threads)
    else:
        pool = None
    for filename in filenames:
        print(filename)
        case, name = os.path.basename(filename).split(".")[0].split("_", 1)
        func = partial(augment_image, filename=filename, case1=case, directory=directory,
                       transforms_directory=transforms_directory, name=name, num_threads=1,
                       output_directory=output_directory)
        if pool:
            pool.map(func, subjects)
        else:
            for subject in subjects:
                func(subject)


def get_subjects():
    return [os.path.basename(f).split("_")[0] for f in
            glob.glob("/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/*_ct.nii.gz")]


def get_filename(case, name, directory="/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data"):
    return os.path.join(directory, "{case}_{name}.nii.gz".format(case=case, name=name))


def augment_image(case2, filename, case1, directory, transforms_directory, output_directory, name, num_threads=1):
    output_filename = os.path.join(output_directory,
                                   "sub-{}_space-{}_{}.nii.gz".format(case1, case2, name))
    if "labelmap" in name:
        interpolation = "NearestNeighbor"
        args = "-u uchar"
    else:
        interpolation = "Linear"
        args = ""
    transforms, invert_transform_flags = get_transforms(case1=case1, case2=case2, directory=transforms_directory)

    if not os.path.exists(output_filename):
        if case1 == case2:
            shutil.copy(filename, output_filename)
        else:
            if not os.path.exists(os.path.dirname(output_filename)):
                os.makedirs(os.path.dirname(output_filename))
            apply_transforms(filename, get_filename(case2, directory=directory, name=name),
                             output_filename=output_filename,
                             transforms=transforms, invert_transform_flags=invert_transform_flags,
                             num_threads=num_threads, interpolation=interpolation, args=args)


def apply_transforms(input_filename, reference_filename, transforms, output_filename,
                     interpolation="NearestNeighbor", args="-u uchar", num_threads=1,
                     invert_transform_flags=None):
    print(input_filename, "-->", reference_filename)
    if invert_transform_flags is None:
        invert_transform_flags = [False for t in transforms]
    cmd = ApplyTransforms(transforms=transforms, input_image=input_filename, output_image=output_filename,
                          reference_image=reference_filename, interpolation=interpolation,
                          num_threads=num_threads,
                          invert_transform_flags=invert_transform_flags)
    if args:
        cmd.inputs.args = args
    print(cmd.cmdline)
    cmd.run()
    return cmd.output_spec().output_image


def all_transforms_exist(prefix, output_directory):
    return all_files_exist(get_prefix_transforms(prefix, output_directory))


def all_files_exist(filenames):
    return all([os.path.exists(fn) for fn in filenames])


def get_transforms(case1, case2, directory):
    # the prefixes are backwards because I performed the registrations backwards
    transforms = get_prefix_transforms("{}_to_{}".format(case2, case1), directory=directory)
    if all_files_exist(transforms):
        return [transforms[1], transforms[0]], [False, False]
    else:
        transforms = get_prefix_transforms("{}_to_{}".format(case1, case2), directory=directory)
        return [transforms[0], transforms[2]], [True, False]


def get_prefix_transforms(prefix, directory):
    affine_filename = os.path.join(directory, prefix + "0GenericAffine.mat")
    warp_filename = os.path.join(directory, prefix + "1Warp.nii.gz")
    inverse_warp_filename = os.path.join(directory, prefix + "1InverseWarp.nii.gz")
    return affine_filename, warp_filename, inverse_warp_filename


def run_registrations():
    ct = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/{}_ct.nii.gz"
    t1 = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/{}_t1.nii.gz"
    t2 = "/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/{}_t2.nii.gz"
    subjects = [os.path.basename(f).split("_")[0] for f in
                glob.glob("/work/aizenberg/dgellis/MICCAI_ABCs_2020/ABCs_training_data/*_ct.nii.gz")]
    for i in range(len(subjects)):
        s1 = subjects[i]
        for s2 in subjects[(i + 1):]:
            print(s1, s2)
            bn = "{}_to_{}".format(s1, s2)
            cmd = ['antsRegistrationSyNQuick.sh',
                             '-f',
                             ct.format(s1),
                             '-f',
                             t1.format(s1),
                             '-f',
                             t2.format(s1),
                             '-m',
                             ct.format(s2),
                             '-m',
                             t1.format(s2),
                             '-m',
                             t2.format(s2),
                             '-n',
                             '32',
                             '-d',
                             '3',
                             '-o',
                             bn]
            print(" ".join(cmd))
            subprocess.call(cmd)


if __name__ == "__main__":
    main()
