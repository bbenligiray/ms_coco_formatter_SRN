import os
import h5py
import zipfile

import numpy as np
from PIL import Image
from scipy.io import loadmat

from calculate_mean import calculate_mean


def main():
	# clean up
	"""for dir_name in ['train2014', 'val2014']:
		if os.path.isdir(dir_name):
			shutil.rmtree(dir_name)

	dataset_links = ['http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
					'http://msvocds.blob.core.windows.net/coco2014/val2014.zip']
	
	for dataset_link in dataset_links:
		file_name = dataset_link.split('/')[-1]
		# download the zip file if it is not there
		if not os.path.isfile(file_name):
			os.system('wget -t0 -c ' + dataset_link)
		#extract the downloaded file
		with zipfile.ZipFile(file_name) as zf:
			zf.extractall()"""
	
	with open('cats') as f:
		cats = f.read().split('\n')
	data_types = ['train', 'val']

	f_out = h5py.File('ms_coco.h5', 'w')
	dt_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
	dt_str = h5py.special_dtype(vlen=str)

	# write the data types and the cats to the .h5 file
	data_types_h = f_out.create_dataset('data_types', (len(data_types),), dtype=dt_str)
	for ind_data_type, data_type in enumerate(data_types):
		data_types_h[ind_data_type] = data_type
	cats_h = f_out.create_dataset('cats', (len(cats),), dtype=dt_str)
	for ind_cat, cat in enumerate(cats):
		cats_h[ind_cat] = cat

	for data_type in data_types:
		# read image names
		if data_type == 'train':
			with open(os.path.join('coco', 'coco_train_imglist.txt')) as f:
				image_names_raw = f.read().splitlines()
			with open(os.path.join('coco', 'coco_train_label.txt')) as f:
				labels_raw = f.read().splitlines()
		elif data_type == 'val':
			with open(os.path.join('coco', 'coco_test_imglist.txt')) as f:
				image_names_raw = f.read().splitlines()
			with open(os.path.join('coco', 'coco_test_label.txt')) as f:
				labels_raw = f.read().splitlines()

		image_names = []
		labels = []
		for ind, label_raw in enumerate(labels_raw):
			image_names.append(image_names_raw[ind].split()[0])
			np_labels = np.fromstring(label_raw, dtype=np.int, sep=' ')
			labels.append(np_labels)

		# write to the .h5 file
		image_h = f_out.create_dataset(data_type + '_images', (len(image_names),), dtype=dt_uint8)
		name_h = f_out.create_dataset(data_type + '_image_names', (len(image_names),), dtype=dt_str)
		shape_h = f_out.create_dataset(data_type + '_image_shapes', (len(image_names), 3), dtype=np.int)
		label_h = f_out.create_dataset(data_type + '_labels', (len(image_names), len(cats)), dtype=np.int)

		for ind, image_name in enumerate(image_names):
			image = Image.open(image_name)
			np_image = np.array(image)

			# if the image is grayscale, repeat its channels to make it RGB
			if len(np_image.shape) == 2:
				np_image = np.repeat(np_image[:, :, np.newaxis], 3, axis=2)

			image_h[ind] = np_image.flatten()
			name_h[ind] = image_name
			shape_h[ind] = np_image.shape
			label_h[ind] = labels[ind]

	f_out.close()

	calculate_mean()

	# show random images to test
	f_in = h5py.File('ms_coco.h5', 'r')
	cats_h = f_in['cats']
	data_types_h = f_in['data_types']
	while True:
		ind_data_type = np.random.randint(0, len(data_types_h))
		data_type = data_types_h[ind_data_type]

		image_h = f_in[data_type + '_images']
		name_h = f_in[data_type + '_image_names']
		shape_h = f_in[data_type + '_image_shapes']
		label_h = f_in[data_type + '_labels']

		ind_image = np.random.randint(0, len(image_h))

		np_image = np.reshape(image_h[ind_image], shape_h[ind_image])
		image = Image.fromarray(np_image, 'RGB')
		image.show()

		print('Image type: ' + data_type)
		print('Image name: ' + name_h[ind_image])
		for ind_cat, cat in enumerate(cats_h):
			if label_h[ind_image][ind_cat] == 1:
				print cat
		raw_input("...")


if __name__ == "__main__":
    main()