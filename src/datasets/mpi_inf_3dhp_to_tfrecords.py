""" Convert MPI_INF_3DHP to TFRecords """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs

import numpy as np
import random
import tensorflow as tf

from .common import convert_to_example_wmosh, ImageCoder, resize_img
from .mpi_inf_3dhp.read_mpi_inf_3dhp import get_paths, read_mat, mpi_inf_3dhp_to_lsp_idx, read_camera

tf.app.flags.DEFINE_string('data_directory', '/scratch1/storage/mpi_inf_3dhp/',
						   'data directory: top of mpi-inf-3dhp')
tf.app.flags.DEFINE_string('output_directory',
						   '/scratch1/projects/tf_datasets/mpi_inf_3dhp/',
						   'Output data directory')

tf.app.flags.DEFINE_string('split', 'train', 'train or trainval')
tf.app.flags.DEFINE_integer('train_shards', 500,
							'Number of shards in training TFRecord files.')

FLAGS = tf.app.flags.FLAGS
MIN_VIS_PTS = 8  # This many points must be within the image.

# To go to h36m joints:
# training joints have 28 joints
# test joints are 17 (H3.6M subset in CPM order)
joint_idx2lsp, test_idx2lsp = mpi_inf_3dhp_to_lsp_idx()


def sample_frames(gt3ds):
	use_these = np.zeros(gt3ds.shape[0], bool)
	# Always use_these first frame.
	use_these[0] = True
	prev_kp3d = gt3ds[0]
	for itr, kp3d in enumerate(gt3ds):
		if itr > 0:
			# Check if any joint moved more than 200mm.
			if not np.any(np.linalg.norm(prev_kp3d - kp3d, axis=1) >= 200):
				continue
		use_these[itr] = True
		prev_kp3d = kp3d

	return use_these


def get_all_data(base_dir, sub_id, seq_id, cam_ids, all_cam_info):
	img_dir, anno_path = get_paths(base_dir, sub_id, seq_id)
	# shiyu
	#img_dir =  ./all_data/mpi_inf_3dhp/S1/Seq1/imageFrames

	# Get data for all cameras.
	frames, _, annot2, annot3 = read_mat(anno_path)

	all_gt2ds, all_gt3ds, all_img_paths = [], [], []
	all_cams = []
	for cam_id in cam_ids:
		base_path = join(img_dir, 'video_%d' % cam_id, 'frame_%06d.jpg')
		num_frames = annot2[cam_id].shape[0]
		gt2ds = annot2[cam_id].reshape(num_frames, -1, 2)
		gt3ds = annot3[cam_id].reshape(num_frames, -1, 3)
		# Convert N x 28 x . to N x 14 x 2, N x 14 x 3
		gt2ds = gt2ds[:, joint_idx2lsp, :]
		gt3ds = gt3ds[:, joint_idx2lsp, :]
		img_paths = [base_path % (frame + 1) for frame in frames]
		if gt3ds.shape[0] != len(img_paths):
			print('Not same paths?')
			import ipdb
			ipdb.set_trace()
		use_these = sample_frames(gt3ds)
		all_gt2ds.append(gt2ds[use_these])
		all_gt3ds.append(gt3ds[use_these])
		K = all_cam_info[cam_id]
		flength = 0.5 * (K[0, 0] + K[1, 1])
		ppt = K[:2, 2]
		flengths = np.tile(flength, (np.sum(use_these), 1))
		ppts = np.tile(ppt, (np.sum(use_these), 1))
		cams = np.hstack((flengths, ppts))
		all_cams.append(cams)
		all_img_paths += np.array(img_paths)[use_these].tolist()

	all_gt2ds = np.vstack(all_gt2ds)
	all_gt3ds = np.vstack(all_gt3ds)
	all_cams = np.vstack(all_cams)

	return all_img_paths, all_gt2ds, all_gt3ds, all_cams


def check_good(image, gt2d):
	h, w, _ = image.shape

	x_in = np.logical_and(gt2d[:, 0] < w, gt2d[:, 0] >= 0)
	y_in = np.logical_and(gt2d[:, 1] < h, gt2d[:, 1] >= 0)

	ok_pts = np.logical_and(x_in, y_in)

	return np.sum(ok_pts) >= MIN_VIS_PTS


def add_to_tfrecord(im_path,
					gt2d,
					gt3d,
					cam,
					coder,
					writer,
					model=None,
					sub_path=None):
	"""
	gt2d is 14 x 2 (lsp order)
	gt3d is 14 x 3
	cam is (3,)
	returns:
	  success = 1 if this is a good image
	  0 if most of the kps are outside the image
	"""
	# Read image
	if not exists(im_path):
		print('!!--%s doesnt exist! Skipping..--!!' % im_path)

		return ('0')
	with tf.gfile.FastGFile(im_path, 'rb') as f:
		image_data = f.read()
	image = coder.decode_jpeg(coder.png_to_jpeg(image_data))
	assert image.shape[2] == 3

	good = check_good(image, gt2d)
	if not good:
		print("not good")
		if FLAGS.split == 'test':
			print('Why no good?? shouldnt happen')
			import ipdb
			ipdb.set_trace()
		return ('0')

	# All kps are visible in mpi_inf_3dhp.
	min_pt = np.min(gt2d, axis=0)
	max_pt = np.max(gt2d, axis=0)
	person_height = np.linalg.norm(max_pt - min_pt)
	center = (min_pt + max_pt) / 2.
	scale = 150. / person_height

	image_scaled, scale_factors = resize_img(image, scale)
	height, width = image_scaled.shape[:2]
	joints_scaled = np.copy(gt2d)
	joints_scaled[:, 0] *= scale_factors[0]
	joints_scaled[:, 1] *= scale_factors[1]
	center_scaled = np.round(center * scale_factors).astype(np.int)
	# scale camera: Flength, px, py
	cam_scaled = np.copy(cam)
	cam_scaled[0] *= scale
	cam_scaled[1] *= scale_factors[0]
	cam_scaled[2] *= scale_factors[1]

	# Crop 300x300 around the center
	margin = 150
	start_pt = np.maximum(center_scaled - margin, 0).astype(int)
	end_pt = (center_scaled + margin).astype(int)
	end_pt[0] = min(end_pt[0], width)
	end_pt[1] = min(end_pt[1], height)
	image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
		0], :]
	# Update others too.
	joints_scaled[:, 0] -= start_pt[0]
	joints_scaled[:, 1] -= start_pt[1]
	center_scaled -= start_pt
	# Update principal point:
	cam_scaled[1] -= start_pt[0]
	cam_scaled[2] -= start_pt[1]
	height, width = image_scaled.shape[:2]

	# Fix units: mm -> meter
	gt3d = gt3d / 1000.

	# Encode image:
	image_data_scaled = coder.encode_jpeg(image_scaled)
	label = np.vstack([joints_scaled.T, np.ones((1, joints_scaled.shape[0]))])
	# pose and shape is not existent.
	pose, shape = None, None
	# example = convert_to_example_wmosh(
	# 	image_data_scaled, im_path, height, width, label, center_scaled, gt3d,
	# 	pose, shape, scale_factors, start_pt, cam_scaled)
	# writer.write(example.SerializeToString())
	# print('successfully written')
	return ('1', image_data_scaled, im_path, height, width, label, center_scaled, gt3d, pose, shape, scale_factors, start_pt, cam_scaled)
	# return True

def save_to_tfrecord(out_name, im_paths_list, gt2ds_list, gt3ds_list, cams_list, num_shards):
	coder = ImageCoder()
	i = 0
	# Count on shards
	fidx = 0
	# Count failures
	num_bad = 0
	## shiyu add multi-view
	num_views = len(im_paths_list) 

	# we assume there's always a main view
	im_paths_2 = [] # view 2 init
	gt2ds_2 = []
	gt3ds_2 = []
	cams_2 = []

	im_paths_3 = [] # view 3 init
	gt2ds_3 = []
	gt3ds_3 = []
	cams_3 = []

	im_paths_4 = [] # view 4 init
	gt2ds_4 = []
	gt3ds_4 = []
	cams_4 = []
	
	# init 
	im_paths_1 = im_paths_list[0] # main view
	gt2ds_1 = gt2ds_list[0]
	gt3ds_1 = gt3ds_list[0]
	cams_1 = cams_list[0]
	
	im_paths_2 = im_paths_list[1] # view 2
	gt2ds_2 = gt2ds_list[1]
	gt3ds_2 = gt3ds_list[1]
	cams_2 = cams_list[1]
	
	im_paths_3 = im_paths_list[2] # view 3
	gt2ds_3 = gt2ds_list[2]
	gt3ds_3 = gt3ds_list[2]
	cams_3 = cams_list[2]

	im_paths_4 = im_paths_list[3] # view 4
	gt2ds_4 = gt2ds_list[3]
	gt3ds_4 = gt3ds_list[3]
	cams_4 = cams_list[3]


	while i < len(im_paths_1):
		tf_filename = out_name % fidx
		print('Starting tfrecord file %s' % tf_filename)
		with tf.python_io.TFRecordWriter(tf_filename) as writer:
			j = 0
			while i < len(im_paths_1) and j < num_shards:
				if i % 100 == 0:
					print('Reading img %d/%d' % (i, len(im_paths_1)))
				# success = add_to_tfrecord(im_paths_1[i], gt2ds_1[i], gt3ds_1[i], cams_1[i], coder, writer)
				
				# random shuffle views shiyu 
				
				random_list = random.sample(range(0, 4), 4)
				# print(random_list)
				im_paths_1 = im_paths_list[random_list[0]] # main view
				gt2ds_1 = gt2ds_list[random_list[0]]
				gt3ds_1 = gt3ds_list[random_list[0]]
				cams_1 = cams_list[random_list[0]]
				
				im_paths_2 = im_paths_list[random_list[1]] # view 2
				gt2ds_2 = gt2ds_list[random_list[1]]
				gt3ds_2 = gt3ds_list[random_list[1]]
				cams_2 = cams_list[random_list[1]]
				
				im_paths_3 = im_paths_list[random_list[2]] # view 3
				gt2ds_3 = gt2ds_list[random_list[2]]
				gt3ds_3 = gt3ds_list[random_list[2]]
				cams_3 = cams_list[random_list[2]]

				im_paths_4 = im_paths_list[random_list[3]] # view 4
				gt2ds_4 = gt2ds_list[random_list[3]]
				gt3ds_4 = gt3ds_list[random_list[3]]
				cams_4 = cams_list[random_list[3]]

				#####
				# view 1 shiyu 
				return_tuple_1 = add_to_tfrecord(im_paths_1[i], gt2ds_1[i], gt3ds_1[i], cams_1[i], coder, writer)
				# check data quality, if one view is not good, then we drop all of them
				if len(return_tuple_1)==1:
					i+=1
					num_bad+=1
					print(str(num_bad)," data dropped due to bad quality, view 1")
					continue
				# success, image_data_scaled, im_path, height, width, label, center_scaled, gt3d, pose, shape, scale_factors, start_pt, cam_scaled
				success_1 = return_tuple_1[0]
				[image_data_scaled_1, im_path_1, height_1, width_1, label_1, center_scaled_1, 
				gt3d_1, pose_1, shape_1, scale_factors_1, start_pt_1, cam_scaled_1] = return_tuple_1[1:] 
				# print("haven't done anything here")
				
				if num_views >=2:
				# view 2 shiyu 
					return_tuple_2 = add_to_tfrecord(im_paths_2[i], gt2ds_2[i], gt3ds_2[i], cams_2[i], coder, writer)
					# check data quality, if one view is not good, then we drop all of them
					if len(return_tuple_2)==1:
						i+=1
						num_bad+=1
						print(str(num_bad)," data dropped due to bad quality, view 2")
						continue
					# success, image_data_scaled, im_path, height, width, label, center_scaled, gt3d, pose, shape, scale_factors, start_pt, cam_scaled
					success_2 = return_tuple_2[0]
					[image_data_scaled_2, im_path_2, height_2, width_2, label_2, center_scaled_2, 
					gt3d_2, pose_2, shape_2, scale_factors_2, start_pt_2, cam_scaled_2] = return_tuple_2[1:]
				else:
					success_2 = '0';
					image_data_scaled_2, im_path_2, height_2, width_2, label_2, center_scaled_2, 
					gt3d_2, pose_2, shape_2, scale_factors_2, start_pt_2, cam_scaled_2 = None

				if num_views >=3:
				# view 3 shiyu 
					return_tuple_3 = add_to_tfrecord(im_paths_3[i], gt2ds_3[i], gt3ds_3[i], cams_3[i], coder, writer)
					# check data quality, if one view is not good, then we drop all of them
					if len(return_tuple_3)==1:
						i+=1
						num_bad+=1
						print(str(num_bad)," data dropped due to bad quality, view 3")
						continue
					# success, image_data_scaled, im_path, height, width, label, center_scaled, gt3d, pose, shape, scale_factors, start_pt, cam_scaled
					success_3 = return_tuple_3[0]
					# print(len(return_tuple_3))
					[image_data_scaled_3, im_path_3, height_3, width_3, label_3, center_scaled_3, gt3d_3, pose_3, shape_3, scale_factors_3, start_pt_3, cam_scaled_3] = return_tuple_3[1:]
				else:
					success_3 = '0';
					image_data_scaled_3, im_path_3, height_3, width_3, label_3, center_scaled_3, 
					gt3d_3, pose_3, shape_3, scale_factors_3, start_pt_3, cam_scaled_3 = None

				if num_views >=4:
				# view 4 shiyu 
					return_tuple_4 = add_to_tfrecord(im_paths_4[i], gt2ds_4[i], gt3ds_4[i], cams_4[i], coder, writer)
					
					if len(return_tuple_4)==1:
						i+=1
						num_bad+=1
						print(str(num_bad), " data dropped due to bad quality, view 4")
						continue
					# success, image_data_scaled, im_path, height, width, label, center_scaled, gt3d, pose, shape, scale_factors, start_pt, cam_scaled
					success_4 = return_tuple_4[0]
					[image_data_scaled_4, im_path_4, height_4, width_4, label_4, center_scaled_4, 
					gt3d_4, pose_4, shape_4, scale_factors_4, start_pt_4, cam_scaled_4] = return_tuple_4[1:]
				else:
					success_4 = '0';
					image_data_scaled_4, im_path_4, height_4, width_4, label_4, 
					center_scaled_4, gt3d_4, pose_4, shape_4, scale_factors_4, start_pt_4, cam_scaled_4 = None

				example = convert_to_example_wmosh(
					image_data_scaled_1, image_data_scaled_2, image_data_scaled_3, image_data_scaled_4, 
					im_path_1, height_1, height_2,height_3,height_4,width_1, width_2,width_3,width_4, label_1, label_2, label_3, label_4, center_scaled_1, center_scaled_2,center_scaled_3,center_scaled_4,gt3d_1,
					pose_1, shape_1, scale_factors_1, start_pt_1, cam_scaled_1)
				writer.write(example.SerializeToString())
				
				i += 1
				j += 1
				# if success_1=='1':
				# 	j += 1
				# 	if i-1 % 100 == 0:
				# 		print('everything good')
				# else:
				# 	num_bad += 1

		fidx += 1

	print('Done, wrote to %s, num skipped %d' % (out_name, num_bad))


def process_mpi_inf_3dhp_train(data_dir, out_dir, is_train=False):
	if is_train:
		out_dir = join(out_dir, 'train')
		print('!train set!')
		# shiyu, make it toy
		# sub_ids = range(1, 8)  # No S8!
		sub_ids = [1,3,4,5,6,7]  # only s1
		# shiyu
		# seq_ids = range(1, 3)
		seq_ids = range(1, 3)
		# shiyu 
		# cam_ids = [0, 1, 2, 4, 5, 6, 7, 8]
		# cam_ids = [0,1,2,4]
		cam_ids = [[0],[2],[5],[8]]
	else:  # Full set!!
		out_dir = join(out_dir, 'trainval')
		print('doing the full train-val set!')
		sub_ids = range(1, 9)
		seq_ids = range(1, 3)
		cam_ids = [0, 1, 2, 4, 5, 6, 7, 8]

	if not exists(out_dir):
		makedirs(out_dir)

	out_path = join(out_dir, 'train_%04d.tfrecord')
	num_shards = FLAGS.train_shards
	all_cam_info = read_camera(data_dir)
	# Load all data & shuffle it,,
	# shiyu main view
	# shiyu list to hold all views
	all_gt2ds_list, all_gt3ds_list, all_img_paths_list = [], [], []
	all_cams_list = []
	# shiyu set number of views
	num_views = len(cam_ids)
	for k in range(len(cam_ids)):
		all_gt2ds, all_gt3ds, all_img_paths = [], [], []
		all_cams = []

		for sub_id in sub_ids:
			for seq_id in seq_ids:
				print('collecting S%d, Seq%d, view%d'% (sub_id, seq_id, cam_ids[k][0]))
				# Collect all data for each camera.
				# img_paths: N list
				# gt2ds/gt3ds: N x 17 x 2, N x 17 x 3
				img_paths, gt2ds, gt3ds, cams = get_all_data(
					data_dir, sub_id, seq_id, cam_ids[k], all_cam_info)
			
				all_img_paths += img_paths
				all_gt2ds.append(gt2ds)
				all_gt3ds.append(gt3ds)
				all_cams.append(cams)	

		all_gt2ds_list.append(np.vstack(all_gt2ds))
		all_gt3ds_list.append(np.vstack(all_gt3ds))
		all_cams_list.append(np.vstack(all_cams))
		all_img_paths_list.append(all_img_paths)

	# Now shuffle it all.
	# shiyu don't shuffle 

	# shuffle_id = np.random.permutation(len(all_img_paths))
	# all_img_paths = np.array(all_img_paths)[shuffle_id]
	# all_gt2ds = all_gt2ds[shuffle_id]
	# all_gt3ds = all_gt3ds[shuffle_id]
	# all_cams = all_cams[shuffle_id]
	assert (len(all_img_paths_list) == len(all_gt2ds_list) == len(all_gt3ds_list) == len(all_cams_list))
	## shiyu don't shuffle
	# print(all_img_paths_list[0][5034])
	# print(all_img_paths_list[1][5034])
	# print("+++++++++++++++")
	# print(len(all_img_paths_list))
	save_to_tfrecord(out_path, all_img_paths_list, all_gt2ds_list, all_gt3ds_list, all_cams_list,
					 num_shards)


def main(unused_argv):
	print('Saving results to %s' % FLAGS.output_directory)

	if not exists(FLAGS.output_directory):
		makedirs(FLAGS.output_directory)

	if FLAGS.split == 'train' or FLAGS.split == 'trainval':
		is_train = FLAGS.split == 'train'
		process_mpi_inf_3dhp_train(
			FLAGS.data_directory, FLAGS.output_directory, is_train=is_train)
	else:
		print('Unknown split %s' % FLAGS.split)
		import ipdb
		ipdb.set_trace()


if __name__ == '__main__':
	tf.app.run()
