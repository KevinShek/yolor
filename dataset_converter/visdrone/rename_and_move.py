import os 

path = "./VisDrone2019-VID-val/sequences"
files = os.listdir(path)

for sub_files in files:
    sub_files_path = os.path.join(path, sub_files)
    images_names = os.listdir(sub_files_path)
    # the following code will rename the file and relocate it to a different path which is what the 2nd argument is for.
    for file in images_names:
        os.rename(os.path.join(sub_files_path, file), os.path.join(path, f"{sub_files}_{file}"))