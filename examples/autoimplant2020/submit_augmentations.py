import subprocess
import glob
import os
import io
import time
import copy


def check_queue_length():
    proc = subprocess.Popen(["squeue", "-u", "dgellis"], stdout=subprocess.PIPE)
    return len(list(io.TextIOWrapper(proc.stdout, encoding="utf-8")))


def wait_for_long_queue(sleeping_time=60, limit=1000):
    while check_queue_length() > limit:
        time.sleep(sleeping_time)


def main():
    skulls = glob.glob("/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set/complete_skull/*.nii.gz")
    cases1 = sorted([os.path.basename(s).split(".")[0] for s in skulls])
    cases2 = copy.copy(cases1)
    template = os.path.join("/work/aizenberg/dgellis/MICCAI_Implant_2020/training_set/registrations",
                            "augmented_{name}/sub-{case1}_space-{case2}_{name}.nii.gz")
    for i, case1 in enumerate(cases1):
        for case2 in cases2[(i+1):]:
            outputs_exist = list()
            for name in ("defective_skull", "implant"):
                outputs_exist.append(os.path.exists(template.format(case1=case1, case2=case2, name=name)))
                outputs_exist.append(os.path.exists(template.format(case1=case2, case2=case1, name=name)))
            if not all(outputs_exist):
                wait_for_long_queue()
                print("Submitting:", case1, "to", case2)
                subprocess.call(["sbatch", "/home/aizenberg/dgellis/fCNN/autoimplant/augmentation_script.sh", case1,
                                 case2])
            else:
                print("Outputs already exist:", case1, "to", case2)


if __name__ == "__main__":
    main()
