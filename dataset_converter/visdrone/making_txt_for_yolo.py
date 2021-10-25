import os, shutil


def making_text_for_yolo(folder_name, input_img_folder):
    image_list = os.listdir(input_img_folder)

    image_path = []

    for image_name in image_list:
        adding_a_dot_to_image_path = f"../../visdrone/{input_img_folder}/{image_name}"
        image_path.append(adding_a_dot_to_image_path)

    def saving_location_of_image_and_label(image_path):
        with open(f"{folder_name}_images_darknet.txt", "a") as f:
            for i in range(len(image_path)):
                f.write(f"{str(image_path[i])}\n")

    saving_location_of_image_and_label(image_path)


def move_files(files, path, output_path):
    for sub_files in files:
        sub_files_path = os.path.join(path, sub_files)
        # the following code will rename the file and relocate it to a different path which is what the 2nd argument is for.
        os.rename(os.path.join(sub_files_path, sub_files), os.path.join(output_path, f"{sub_files}"))


def copy_files(files, path, output_path):
    for sub_file in files:
        file_path = os.path.join(path, sub_file)
        shutil.copy(file_path, output_path)


def remove_files(files, path):
    for sub_file in files:
        file_path = os.path.join(path, sub_file)
        os.remove(file_path)


def find_text_files(path):
    text_file = []
    files = os.listdir(path)
    for file in files:
        if file.endswith(".txt"):
            text_file.append(file)

    return text_file


def main():
    folder_names = ["VisDrone2019-DET-test-dev", "VisDrone2019-DET-train", "VisDrone2019-DET-val"]
    for folder_name in folder_names:
        input_img_folder = f'{folder_name}/images'
        input_ann_folder = f'{folder_name}/labels'
        # making_text_for_yolo(folder_name, input_img_folder)

        ann_list = os.listdir(input_ann_folder)

        # making text files for darknet
        # making_text_for_yolo(folder_name, input_img_folder)

        # for moving into the image folder
        copy_files(ann_list, input_ann_folder, input_img_folder)

        # to find and delete files
        # files = find_text_files(input_img_folder)
        # remove_files(files, input_img_folder)


if __name__ == "__main__":
    main()