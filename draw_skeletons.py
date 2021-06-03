import json
import os
import glob
import math
import numpy as np
import cv2
from preprocessing import *
from subprocess import call
import argparse


def __draw_limbs(image, keypoints_x, keypoints_y):
	chest = [(1,2), (1,5), (1,8), (2,3), (5,6), (6,7), (3,4)]
	head_neck = [(0,15), (0,16), (0,1)]
	legs_feet = [(8,9), (8,12), (9,10), (12,13), (13,14), (10,11), (11,18), (14,17)]

	h, w = image.shape[:2]
	#15 16 20 21 23 24 

	for joint in head_neck:
		cv2.line(image, (keypoints_x[joint[0]] + int(w / 2), keypoints_y[joint[0]] + int(h / 2)),
			(keypoints_x[joint[1]] + int(w / 2), keypoints_y[joint[1]] + int(h / 2)), color=(10,10,255), thickness=1)

	for joint in chest:
		cv2.line(image, (keypoints_x[joint[0]] + int(w / 2), keypoints_y[joint[0]] + int(h / 2)),
			(keypoints_x[joint[1]] + int(w / 2), keypoints_y[joint[1]] + int(h / 2)), color=(0,255,255), thickness=1)

	for joint in legs_feet:
		cv2.line(image, (keypoints_x[joint[0]] + int(w / 2), keypoints_y[joint[0]] + int(h / 2)),
			(keypoints_x[joint[1]] + int(w / 2), keypoints_y[joint[1]] + int(h / 2)), color=(255,162,80), thickness=1)


def draw_skeleton(skeleton_keypoints, batch_index, frame_index, path="Batches"):
	keypoints_x = []
	keypoints_y = []

	for i in range(len(skeleton_keypoints)):
		if i % 2 == 0:
			keypoints_x.append(int(skeleton_keypoints[i]*200))
		else:
			keypoints_y.append(int(skeleton_keypoints[i]*300))

	image = np.zeros((2304, 1296), np.uint8)

	# Trasformation into the image center
	# x = np.array(keypoints_x)
	# y = np.array(keypoints_y)

	# Calculate centroids
	# cx = int(np.sum(x) / np.count_nonzero(x))
	# cy = int(np.sum(y) / np.count_nonzero(y))

	h, w = image.shape[:2]
	# Since the origin (0,0) is in the top-left corner, for viewing purpose we traslate once again, this time into the middle (h/2,w/2)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	# Center of gravity
	#cv2.circle(image, (cx, cy), radius=4, color=(0, 255, 0), thickness=3)
	
	# Draw centered skeleton keypoints
	for i in range(len(keypoints_x)):
		color = (0,0,0)
		if i == 0 or i == 15 or i == 16:											# Head
			color = (10,10,255)
		elif i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7:	# Chest
			color = (0,255,255)
		else: 																		# Lower body
			color = (255,162,80)
		cv2.circle(image, (keypoints_x[i] + int(w / 2), keypoints_y[i] + int(h / 2)), radius=3, color=color, thickness=3)

	# Draw limbs between joints
	__draw_limbs(image, keypoints_x, keypoints_y)
	
	# Write frames
	sub_dir = os.path.join(path,str(batch_index))
	if not os.path.exists(sub_dir):
		os.makedirs(sub_dir)
	cv2.imwrite(sub_dir + "/frame_" + str(frame_index) + ".png", image)


# def trasformation(skeleton_keypoints):
#     # Skeletons given to the model are always centered in the origin (0,0). In such a way we make the network position-invariant
#     keypoints_x = []
#     keypoints_y = []
#     new_skeletons = skeleton_keypoints
    
#     for skeleton in new_skeletons:
#         del skeleton[2::3]  # Get rid of the scores
#         cx = 0
#         cy = 0

#         # Calculate the center of the current skeleton as its center of gravity. Precautions are taken so that we
#         # divide by the number of non-null keypoints
#         for i in range(len(skeleton)):
#             if i%2==0:
#                 keypoints_x.append(int(skeleton[i]))
#             else:
#                 keypoints_y.append(int(skeleton[i]))
            
#         x = np.array(keypoints_x)
#         y = np.array(keypoints_y)
#         cx = int(np.sum(x)/np.count_nonzero(x))
#         cy = int(np.sum(y)/np.count_nonzero(y))

#         for i in range(len(skeleton)):
#             if i%2==0:
#                 skeleton[i] -= cx
#             else:
#                 skeleton[i] -= cy

#         keypoints_x = []
#         keypoints_y = []
#     return new_skeletons


def generate_video(path):
	 # Generate a video for each batch of frames
	for folder in os.listdir('./'+path):
		print("Writing video for batch: " + folder)
		call(['ffmpeg', '-framerate', '10', '-i', path + "/" + str(folder) + '/frame_%01d.png',
			  path + "/" + str(folder) + '.avi'])
		print("Video written")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i' 		,'--input'		,type=str   , required=True		,help='Input data folder')
	parser.add_argument('-o' 		,'--output'		,type=str   , required=True		,help='Output data folder')
	args = parser.parse_args()
	data_path = args.input
	save_path = args.output

	skeletons_keypoints = load_data(path=data_path)
	skeletons_keypoints = remove_low_scoring_keypoints(skeletons_keypoints)
	skeletons_keypoints = remove_null_skeleton(skeletons_keypoints)
	skeletons_keypoints = normalization(skeletons_keypoints)
	batched_scheletons = data_windowing(skeletons_keypoints, overlap=1.5)
	print(len(batched_scheletons), "total windows")

	# Write each batch of frames in a separate folder
	for i in range(len(batched_scheletons)):            #index on batches
		print("Writing batch ", i)
		for j in range(len(batched_scheletons[i])):     #index on frames
			draw_skeleton(batched_scheletons[i][j], i, j, save_path)

	generate_video(save_path)