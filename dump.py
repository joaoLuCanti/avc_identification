import os
import shutil

cur_dir = os.getcwd()

for dirs in os.listdir(cur_dir):
    if ".py" in dirs:
        continue
    patient_path = os.path.join(cur_dir, dirs)
    for slices in os.listdir(patient_path):
        if "dcm" not in slices:
            continue
        slice_path = os.path.join(patient_path, slices)
        new_slice = dirs+slices
        new_slice_path = os.path.join(cur_dir,new_slice)
        shutil.copy(slice_path, new_slice_path)