import os
cur_path = os.getcwd()
count = 0
count_patients = 0
for dirs in os.listdir(cur_path):
    if ".py" in dirs:
        continue
    if "dycom" not in dirs:
        continue
    phase_path = os.path.join(cur_path, dirs)
    for patients in os.listdir(phase_path):
        patient_path = os.path.join(phase_path, patients)
        for slices in os.listdir(patient_path):
            count+=1
            count_patients+=1
        print(f"{patients} = {count_patients}")
        count_patients=0
    print(f"{dirs} = {count}")
    count = 0
