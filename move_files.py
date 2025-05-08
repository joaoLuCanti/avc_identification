import os
import sys
import shutil

cur_path = os.getcwd()

split_names = ["train","test"]
class_names = ["1","0"]
new_class_names = ["normal","pathologic"]

normal_class_path = os.path.join(cur_path,new_class_names[0])
patho_class_path = os.path.join(cur_path,new_class_names[1])
                               

os.makedirs(normal_class_path, exist_ok=True)
os.makedirs(patho_class_path, exist_ok=True)

for i in split_names:
    split_path = os.path.join(cur_path, i)
    for j in class_names:
        class_path = os.path.join(split_path, j)
        for slices in os.listdir(class_path):
            slice_path = os.path.join(class_path, slices)
            if j == "1":
                new_slice_path = os.path.join(patho_class_path, slices)
            else:
                new_slice_path = os.path.join(normal_class_path, slices)
            shutil.copy(slice_path, new_slice_path)