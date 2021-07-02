import numpy as np
import os
import matplotlib.image as mpimg
import ntpath
import math
import random
import matplotlib.pyplot as plt
import fa_util as fu
import glob
import cv2


def load_pts(pts_file_path):
    """ load pts file and return points"""
    f = open(pts_file_path, 'r')  # open pts file
    dump = f.readline()  # first line is version
    dump = f.readline()  # second line is number of points
    a, b = dump.split()
    n_points = int(b)
    dump = f.readline()  # third line is bracket

    pts_buff = []
    for x in range(n_points):  # read pts points
        dump = f.readline()
        a, b = dump.split()
        pts_buff.append(float(a))
        pts_buff.append(float(b))
    f.close()
    pts_array = np.asarray(pts_buff)
    pts = pts_array.reshape(n_points, 2)  # make pts to n by 2 array
    return pts


def load_img_pts(img_file_path):
    """ load image and points"""
    # img = mpimg.imread(img_file_path)
    img = cv2.imread(img_file_path)
    # img = Image.open(img_file_path)

    if img.dtype != np.uint8:
        img = img * 255
        img = img.astype(np.uint8)

    if len(img.shape) == 2:                         # when not being 3 channels
        src_h, src_w = img.shape
        channel_in = 1
    else:
        src_h, src_w, channel_in = img.shape

    if channel_in == 1 and fu.channel == 3:
        img_new = np.zeros((src_h, src_w, fu.channel), np.uint8)
        img_new[:, :, 0] = img
        img_new[:, :, 1] = img
        img_new[:, :, 2] = img
        img = img_new
    elif channel_in == 3 and fu.channel == 1:
        img_new = np.zeros((src_h, src_w), np.uint8)
        img_new[:, :] = np.average(img, axis=2)
        img = img_new

    # when using cv2
    img = img[..., ::-1]        # brg to rgb
    head, tail = ntpath.split(img_file_path)
    img_file_name = tail or ntpath.basename(head)
    pts_file_name = img_file_name[0:len(img_file_name) - 3] + 'pts'
    pts_file_path = img_file_path[0:len(img_file_path) - 3] + 'pts'
    pts = load_pts(pts_file_path)
    return img, pts


def get_bounding_box(pts):
    """ return face bounding box from pts,  [x1, y1; x2,y2]"""
    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)
    bounding_box = np.zeros((2, 2))
    bounding_box[0, :] = pts_min
    bounding_box[1, :] = pts_max
    return bounding_box


def get_bounding_box3_square_with_margin(pts):
    """ return squared face bounding box from pts,  [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]"""
    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)
    pts_size = pts_max - pts_min
    face_width = pts_size[0]
    face_height = pts_size[1]
    bounding_box = np.zeros((2, 2), np.float32)         # [[x1, y3], [x3, y3]]
    bounding_box3 = np.zeros((3, 3), np.float32)        # [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    if face_width > face_height:  # bounding box should be square rectangle
        bounding_box[0, 0] = pts_min[0]     # x1
        bounding_box[0, 1] = pts_min[1] - (face_width - face_height) / 2  # y1
        bounding_box[1, 0] = pts_max[0]     # x3
        bounding_box[1, 1] = pts_max[1] + (face_width - face_height) / 2  # y3
        bounding_box[0, :] = bounding_box[0, :] - face_width * 0.1
        bounding_box[1, :] = bounding_box[1, :] + face_width * 0.1
    else:
        bounding_box[0, 0] = pts_min[0] - (face_height - face_width) / 2  # x1
        bounding_box[0, 1] = pts_min[1]  # y1
        bounding_box[1, 0] = pts_max[0] + (face_height - face_width) / 2  # x3
        bounding_box[1, 1] = pts_max[1]  # y3
        bounding_box[0, :] = bounding_box[0, :] - face_height * 0.1
        bounding_box[1, :] = bounding_box[1, :] + face_height * 0.1
    bounding_box3[0, :] = [bounding_box[0, 0], bounding_box[1, 0], bounding_box[1, 0]]    # [x1, x2, x3]
    bounding_box3[1, :] = [bounding_box[0, 1], bounding_box[0, 1], bounding_box[1, 1]]    # [y1, y2, y3]
    bounding_box3[2, :] = [1, 1, 1]                                                       # [1, 1, 1]
    return bounding_box3


def get_output_path(x, out_folder, ext):
    file_name = os.path.basename(x)
    out_file_name = file_name[0:-3] + ext
    return out_folder + '/' + out_file_name


def save_pts(path, pts):
    pts_shape = pts.shape
    f = open(path, 'w')
    f.write('version: 1\n')
    f.write('n_points: %d\n' % pts_shape[0])
    f.write('{\n')
    for i in range(pts_shape[0]):
        f.write('%f %f\n' % (pts[i, 0], pts[i, 1]))
    f.write('}\n')
    f.close()


def measurement(gt_pts, estimated_pts):
    """return pixel error, error normalized by interocular distance, bounding box distance"""
    pts_error = np.linalg.norm(estimated_pts - gt_pts, axis=1)
    error_pixel = np.average(pts_error, axis=0)

    if len(gt_pts) == 68:
        IOD = np.linalg.norm((gt_pts[36, :] - gt_pts[45, :]), axis=0)
    elif len(gt_pts) == 39:
        IOD = 1

    face_box = get_bounding_box(gt_pts)
    pts_size = face_box[1, :] - face_box[0, :]
    box_size = math.sqrt(pts_size[0]*pts_size[1])

    return error_pixel / IOD, error_pixel / box_size


def make_file_list_by_folder(img_folders, img_extension):
    """ make file list by folder path """
    files = []
    for f in img_folders:
        print('Data Path: ' + f)
        for ext in img_extension:
            current_path = f + '/*.' + ext
            current_files = glob.glob(current_path)
            files.extend(current_files)
    return files


def make_file_list_by_text(list_file_name):
    """ make file list by file list txt """
    files = []
    f = open(list_file_name, 'r')
    line = f.readlines()
    f.close()
    n = len(line)
    for x in line:
        img_path = x[0:-1]
        files.append(img_path)
    return files


def make_chunk_set(files, chunk_size):
    img_files = []
    n_samples = len(files)
    print('number of samples: ' + str(n_samples))

    for x in files:
        img_files.append(x)
    end_idx = 0
    image_data_sets = []
    while end_idx < n_samples:  # make chunk sets
        start_idx = end_idx
        end_idx = start_idx + chunk_size
        if end_idx >= n_samples:
            end_idx = n_samples
        current_set = img_files[start_idx:end_idx]
        image_data_sets.append(current_set)
    return image_data_sets


def shifting_pts(pts, pose_idx):
    scale_x_random = random.gauss(1.0, 0.01)
    scale_y_random = random.gauss(1.0, 0.01)
    rotation_random = random.gauss(0.0, math.pi / 48)
    translation_x_random = random.gauss(0.0, 0.05)
    translation_y_random = random.gauss(0.0, 0.05)
    warp_mat_t, warped_pts = fu.get_warp_mat_t_pts2template(pts, pose_idx)
    affine_warp_mat_t = np.array(
        [[scale_x_random * math.cos(rotation_random), -math.sin(rotation_random), 0],
         [math.sin(rotation_random), scale_y_random * math.cos(rotation_random), 0],
         [translation_x_random, translation_y_random, 1]])
    warped_template_pts = np.dot(np.hstack((warped_pts, np.ones((fu.n_points, 1), np.float32))), affine_warp_mat_t)
    pts_shifted = np.dot(warped_template_pts, np.linalg.inv(warp_mat_t))
    return pts_shifted[:, 0:2]


def get_jittered_bounding_box3(face_box3):
    # scale_x_random = random.uniform(0.9, 1.1)
    # scale_y_random = random.uniform(0.9, 1.1)
    # rotation_random = random.uniform(-math.pi/12, math.pi/12)
    # translation_x_random = random.uniform(-0.1, 0.1)
    # translation_y_random = random.uniform(-0.1, 0.1)
    scale_x_random = random.gauss(1.0, 0.01)
    scale_y_random = random.gauss(1.0, 0.01)
    rotation_random = random.gauss(0.0, math.pi/24)
    translation_x_random = random.gauss(0.0, 0.05)
    translation_y_random = random.gauss(0.0, 0.05)

    affine_warp_mat = np.array([[scale_x_random * math.cos(rotation_random), math.sin(rotation_random), translation_x_random],
                         [-math.sin(rotation_random), scale_y_random * math.cos(rotation_random), translation_y_random],
                         [0, 0, 1]], np.float32)

    A = np.array([[-1, 1, 1], [-1, -1, 1], [1, 1, 1]], np.float32)                                  # normalized box
    # inv_A = np.linalg.inv(A)
    # face_box3_3 = np.hstack((face_box3, np.ones((3, 1), np.float32)))
    # W = np.dot(inv_A, face_box3_3)
    jittered_A = np.dot(affine_warp_mat, A)
    W = np.dot(face_box3, np.linalg.inv(A))     # normalized box -> face box
    jittered_boundbox3 = np.dot(W, np.dot(affine_warp_mat, A))
    return jittered_boundbox3


def get_small_jittered_bounding_box3(face_box3):
    # scale_x_random = random.uniform(0.9, 1.1)
    # scale_y_random = random.uniform(0.9, 1.1)
    # rotation_random = random.uniform(-math.pi/12, math.pi/12)
    # translation_x_random = random.uniform(-0.1, 0.1)
    # translation_y_random = random.uniform(-0.1, 0.1)
    scale_x_random = random.gauss(1.0, 0.001)
    scale_y_random = random.gauss(1.0, 0.001)
    rotation_random = random.gauss(0.0, math.pi/48)
    translation_x_random = random.gauss(0.0, 0.01)
    translation_y_random = random.gauss(0.0, 0.01)

    affine_warp_mat = np.array([[scale_x_random * math.cos(rotation_random), math.sin(rotation_random), translation_x_random],
                         [-math.sin(rotation_random), scale_y_random * math.cos(rotation_random), translation_y_random],
                         [0, 0, 1]], np.float32)

    A = np.array([[-1, 1, 1], [-1, -1, 1], [1, 1, 1]], np.float32)                                  # normalized box
    # inv_A = np.linalg.inv(A)
    # face_box3_3 = np.hstack((face_box3, np.ones((3, 1), np.float32)))
    # W = np.dot(inv_A, face_box3_3)
    jittered_A = np.dot(affine_warp_mat, A)
    W = np.dot(face_box3, np.linalg.inv(A))     # normalized box -> face box
    jittered_boundbox3 = np.dot(W, np.dot(affine_warp_mat, A))
    return jittered_boundbox3


def get_pts_move(gt_pts, pts, pose_idx):
    warp_mat_t, warped_pts = fu.get_warp_mat_t_pts2template(pts, pose_idx)
    warped_gt_pts = fu.get_warped_pts(gt_pts, warp_mat_t)
    pts_move = warped_gt_pts - warped_pts
    return pts_move


def measurement_parts(gt_pts, estimated_pts):
    error_parts = np.zeros(fu.n_parts, np.float32)
    pts_error = np.linalg.norm(estimated_pts - gt_pts, axis=1)
    for i in range(fu.n_parts):
        error_part = np.average(pts_error[fu.parts_idx[i]], axis=0)
        error_parts[i] = error_part

    if len(gt_pts) == 68:
        IOD = np.linalg.norm((gt_pts[36, :] - gt_pts[45, :]), axis=0)
    elif len(gt_pts) == 39:
        IOD = 1

    face_box = get_bounding_box(gt_pts)
    pts_size = face_box[1, :] - face_box[0, :]
    box_size = math.sqrt(pts_size[0]*pts_size[1])

    return error_parts / IOD, error_parts / box_size

