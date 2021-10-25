import cv2
import os
import numpy as np

image_det = False
folder_name = "VisDrone2019-VID-train"
xml = False
new_annotation_method = False

if image_det:
	input_img_folder = f'{folder_name}/images'
	input_ann_folder = f'{folder_name}/annotations'
	output_ann_folder = f'{folder_name}/annotations_new'
	output_img_folder = f'{folder_name}/images_new'
else:
	input_img_folder = f'{folder_name}/sequences'
	input_ann_folder = f'{folder_name}/annotations'
	output_ann_folder = f'{folder_name}/annotations_new'
	output_img_folder = f'{folder_name}/sequences_new'

os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_ann_folder, exist_ok=True)


image_list = os.listdir(input_img_folder)
annotation_list = os.listdir(input_ann_folder)

label_dict = {
	"0" : "Ignore",
	"1" : "Pedestrian",
	"2" : "People",
	"3" : "Bicycle",
	"4" : "Car",
	"5" : "Van",
	"6" : "Truck",
	"7" : "Tricycle",
	"8" : "Awning-tricycle",
	"9" : "Bus",
	"10" : "Motor",
	"11" : "Others"
}

thickness = 2
color = (255,0,0)
count = 0

def object_string(label, bbox):
	req_str = '''
	<object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
	'''.format(label, bbox[0], bbox[1], bbox[2], bbox[3])
	return req_str


def convert_box(size, box):
	# Convert VisDrone box to YOLO xywh box
	dw = 1. / size[0]
	dh = 1. / size[1]
	return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


def writing_annotation_txt(img_frame_number):
	file = open(annotation_path, 'r')
	lines = file.readlines()
	annotation_string = ''''''

	for line in lines:
		new_line = line.strip('\n').split(',')
		if int(new_line[0]) == img_frame_number:
			annotation_string = annotation_string + str(line)
	
	annotation_string_final = annotation_string
	return annotation_string_final


def writing_annotations_yolo(annotation_path, image_det, img, img_frame_number):
	file = open(annotation_path, 'r')
	lines = file.readlines()

	for line in lines:
		new_line = line.strip('\n').split(',')
		if int(new_line[0]) == img_frame_number:
			if image_det:
				adjustment_count = 0
			else:
				adjustment_count = 2
			bbox = (int(new_line[0 + adjustment_count]), int(new_line[1 + adjustment_count]), int(new_line[0 + adjustment_count])+int(new_line[2 + adjustment_count]), int(new_line[1 + adjustment_count])+int(new_line[3 + adjustment_count]))
			label = new_line[5 + adjustment_count]
			size = (img.shape[1], img.shape[0])
			converted_bbox = convert_box(size, bbox)
			string_of_anno = f"{label} {converted_bbox[0]} {converted_bbox[1]} {converted_bbox[2]} {converted_bbox[3]}\n"
			annotation_string = annotation_string + string_of_anno

	annotation_string_final = annotation_string
	return annotation_string_final


def writing_annotations_xml(annotation_path, img_file, image_det, img, img_path, img_frame_number):
	annotation_string_init = '''
	<annotation>
	<folder>annotations</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>{}</depth>
	</size>
	<segmented>0</segmented>'''.format(img_file, img_path, img.shape[1], img.shape[0], img.shape[2])

	file = open(annotation_path, 'r')
	lines = file.readlines()

	first_line = True

	for line in lines:
		new_line = line.strip('\n').split(',')
		if int(new_line[0]) == img_frame_number:
			if image_det:
				count = 0
			else:
				count = 2
			bbox = (int(new_line[0 + count]), int(new_line[1 + count]), int(new_line[0 + count])+int(new_line[2 + count]), int(new_line[1 + count])+int(new_line[3 + count]))
			label = label_dict.get(new_line[5 + count])
			req_str = object_string(label, bbox)
			annotation_string_init = annotation_string_init + req_str
		#cv2.rectangle(img, new_coords_min, new_coords_max, color, thickness)
	# cv2.imwrite(output_img_path, img)
	annotation_string_final = annotation_string_init + '</annotation>'
	return annotation_string_final


def output(annotation_string_final, count, path_format):

	with open(path_format, 'a') as f:	# this code would make a new file and append the data into the file
		f.write(annotation_string_final)
		count += 1
		print('[INFO] Completed {} image(s) and annotation(s) pair'.format(count))
	return count


def saving_location_of_image_and_label(image_path, label_path):
	with open(f"{folder_name}_images.txt", "a") as f:
		for i in range(len(image_path)):
			f.write(f"{str(image_path[i])}\n")
	with open(f"{folder_name}_labels.txt", "a") as f:
		for i in range(len(label_path)):
			f.write(f"{str(label_path[i])}\n")


image_path = []
label_path = []

for annotation in annotation_list:
	if image_det:
		annotation_path = os.path.join(os.getcwd(), input_ann_folder, annotation)
		xml_annotation = annotation.split('.txt')[0] + '.xml'
		xml_path = os.path.join(os.getcwd(), output_ann_folder, xml_annotation)
		img_file = annotation.split('.txt')[0] + '.jpg'
		img_path = os.path.join(os.getcwd(), input_img_folder, img_file)
		output_img_path = os.path.join(os.getcwd(), output_img_folder, img_file)
		img = cv2.imread(img_path)
		annotation_string_final = writing_annotations_xml(annotation_path, img_file, image_det, img, img_path)
		output(annotation_string_final, count, xml_path)
	else:
		annotation_name = annotation.split('.txt')[0]
		# output_seq_folder = os.path.join(output_ann_folder, annotation_name) # location where to save the annotations
		# os.makedirs(output_seq_folder, exist_ok=True) # checking to see if the folder exists
		annotation_path = os.path.join(os.getcwd(), input_ann_folder, annotation)
		input_annotation = os.path.join(os.getcwd(), input_img_folder, annotation_name)
		image_list = os.listdir(input_annotation)
		for img_file in image_list:
			img_frame = img_file.split('.jpg')[0]
			img_frame_number = int(img_frame.split("0", 1)[-1])
			img_path = os.path.join(os.getcwd(), input_annotation, img_file)
			img = cv2.imread(img_path)
			if format == "xml":
				annotation_format = img_file.split('.jpg')[0] + '.xml'
				format_path = os.path.join(os.getcwd(), output_ann_folder, f"{annotation_name}_{annotation_format}")
				annotation_string_final = writing_annotations_xml(annotation_path, img_file, image_det, img, img_path)
				adding_a_dot_to_ann_path = f"./{output_ann_folder}/{annotation_name}_{annotation_format}"
			elif format == "yolo":
				annotation_format = img_file.split('.jpg')[0] + '.txt'
				annotation_string_final = writing_annotations_yolo(annotation_path, image_det, img, format_path, image_list)
				adding_a_dot_to_ann_path = f"./{output_ann_folder}/{annotation_name}_{annotation_format}"
			else:
				format_path = os.path.join(os.getcwd(), output_ann_folder, f"{annotation_name}_{img_frame}.txt")
				annotation_string_final = writing_annotation_txt(img_frame_number)

			count = output(annotation_string_final, count, format_path)
			output_img_path = os.path.join(os.getcwd(), output_img_folder, f"{annotation_name}_{img_file}")
			# adding_a_dot_to_image_path = f"./{output_img_folder}/{annotation_name}_{img_file}"
			# image_path.append(output_img_path)
			# label_path.append(adding_a_dot_to_ann_path)
			#cv2.imwrite(output_img_path, img)


# saving_location_of_image_and_label(image_path, label_path)

