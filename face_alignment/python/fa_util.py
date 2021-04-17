import numpy as np
from skimage import transform
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import fa_util_train as fut
from skimage.transform import PiecewiseAffineTransform, warp
from datetime import datetime
import math

init_w = 224
init_h = 224
mesh_w = 224
mesh_h = 224
patch_size = 31
patch_center_x = []     # init by patch_size
patch_center_y = []     # init by patch_size
channel = 3
n_parts = 11
n_points = 68
n_points2 = 136
n_points_ex = 76
n_pose = 3
img_template_scale = 50

# pose
pose_txt = ['front', 'left', 'right']

# parts
PART_IDX_CHIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
PART_IDX_LEYEBROW = [17, 18, 19, 20, 21]
PART_IDX_REYEBROW = [22, 23, 24, 25, 26]
PART_IDX_NOSE = [27, 28, 29, 30, 31, 32, 33, 34, 35]
PART_IDX_LEYE = [36, 37, 38, 39, 40, 41]
PART_IDX_REYE = [42, 43, 44, 45, 46, 47]
PART_IDX_MOUTH = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

PART_IDX_CHIN0 = [0, 1, 2, 3, 4]
PART_IDX_CHIN1 = [5, 6, 7, 8, 9, 10, 11]
PART_IDX_CHIN2 = [12, 13, 14, 15, 16]
PART_IDX_NOSE0 = [27, 28, 29, 30]
PART_IDX_NOSE1 = [31, 32, 33, 34, 35]
PART_IDX_MOUTH0 = [48, 49, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64]
PART_IDX_MOUTH1 = [55, 56, 57, 58, 59, 65, 66, 67]
parts_idx = [PART_IDX_CHIN0, PART_IDX_CHIN1, PART_IDX_CHIN2, PART_IDX_LEYEBROW, PART_IDX_REYEBROW,
             PART_IDX_NOSE0, PART_IDX_NOSE1, PART_IDX_LEYE, PART_IDX_REYE, PART_IDX_MOUTH0, PART_IDX_MOUTH1]
part_txt = ['lchin', 'fchin', 'rchin', 'leyebrow', 'reyebrow', 'unose', 'lnose', 'leye', 'reye', 'umouth', 'lmouth']
part_channel_idx = []
n_part_points = [5, 7, 5, 5, 5, 4, 5, 6, 6, 12, 8]

# mean shapes
mean_shapes_file_path = '../models/mean_shapes.txt'
mean_shapes_2d = []
mean_shapes_3d = []
MESH_2D_MODEL_MEAN_SHAPE_EX = [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]]
mean_shape_ex_outside = []
img_template_mean_shapes = []

# meshes
mean_shapes_mesh_ex = []

# mean appearance
warped_mean_path = ['../models/warped_mean_front.bmp',
                    '../models/warped_mean_left.bmp', '../models/warped_mean_right.bmp']
warped_mean_appearances = []
img_template_parts = []
img_big_template_parts = []

# shape parameters
parameter_U_file_path = ['../models/shape_parameter_U_front.txt',
                         '../models/shape_parameter_U_left.txt', '../models/shape_parameter_U_right.txt']
parameter_s_file_path = ['../models/shape_parameter_s_front.txt',
                         '../models/shape_parameter_s_left.txt', '../models/shape_parameter_s_left.txt']
shape_dim = 15
parameters = []
pinv_parameters = []
parameters_s = []

# scaling
error_scale = 5
parameter_scale = 10
move_scale = 5
img_template_scale = 50
img_big_template_scale = 112


def fu_init():
    # mean shapes
    global mean_shapes_2d
    global mean_shapes_3d
    f = open(mean_shapes_file_path, 'r')
    line = f.readline()  # first line is 3D mean shape
    points = line.split()
    mean_shapes_3d = np.reshape(np.array(points), (68, 3)).astype(np.float32)
    line = f.readline()  # second line is 2D mean shape front
    points = line.split()
    mean_shapes_2d.append(np.reshape(np.array(points), (68, 2)).astype(np.float32))
    line = f.readline()  # second line is 2D mean shape left
    points = line.split()
    mean_shapes_2d.append(np.reshape(np.array(points), (68, 2)).astype(np.float32))
    line = f.readline()  # second line is 2D mean shape right
    points = line.split()
    mean_shapes_2d.append(np.reshape(np.array(points), (68, 2)).astype(np.float32))
    f.close()

    # meshes
    global mean_shapes_mesh_ex
    for p in range(n_pose):
        mean_shape_mesh = np.vstack((mean_shapes_2d[p], np.array(MESH_2D_MODEL_MEAN_SHAPE_EX)))
        mean_shape_mesh[:, 0] = mean_shape_mesh[:, 0] * mesh_w / 2 + mesh_w / 2
        mean_shape_mesh[:, 1] = mean_shape_mesh[:, 1] * mesh_h / 2 + mesh_h / 2
        mean_shapes_mesh_ex.append(mean_shape_mesh)

    global mean_shape_ex_outside
    mean_shape_ex_outside = np.array(MESH_2D_MODEL_MEAN_SHAPE_EX, np.float32)

    # load mean appearance
    global warped_mean_appearances
    warped_mean = cv2.imread(warped_mean_path[0], cv2.IMREAD_COLOR)
    warped_mean = warped_mean[..., ::-1]  # bgr to rgb
    warped_mean_appearances.append(warped_mean)
    warped_mean = cv2.imread(warped_mean_path[1], cv2.IMREAD_COLOR)
    warped_mean = warped_mean[..., ::-1]  # bgr to rgb
    warped_mean_appearances.append(warped_mean)
    warped_mean = cv2.imread(warped_mean_path[2], cv2.IMREAD_COLOR)
    warped_mean = warped_mean[..., ::-1]  # bgr to rgb
    warped_mean_appearances.append(warped_mean)

    # shape parameters
    global parameters
    global pinv_parameters
    global parameters_s
    for i in range(3):
        parameter_all = np.loadtxt(parameter_U_file_path[i], np.float32)  # load parameter
        parameter = parameter_all[:, 0:shape_dim]
        parameters.append(parameter)
        pinv_parameters.append(np.linalg.pinv(parameter))

        parameter_s_all = np.loadtxt(parameter_s_file_path[i], np.float32)  # load parameter
        parameter_s = parameter_s_all[0:shape_dim]
        parameter_s = np.sqrt(parameter_s)
        parameter_s = np.reshape(parameter_s, (shape_dim, 1))
        parameters_s.append(parameter_s)

    # patches
    global img_template_parts
    global patch_center_x
    global patch_center_y
    patch_center_x = math.floor(patch_size/2) + 1
    patch_center_y = math.floor(patch_size/2) + 1
    for i in range(3):
        img_template_mean_shape = mean_shapes_2d[i] * img_template_scale                # for patch
        img_template_mean_shape += img_template_scale
        img_template_parts.append(get_warp3points(img_template_mean_shape))
        img_big_template_mean_shape = mean_shapes_2d[i] * img_big_template_scale                # for big size (224 by 224)
        img_big_template_mean_shape += img_big_template_scale
        img_big_template_parts.append(get_warp3points(img_big_template_mean_shape))

    global part_channel_idx                                                               # make index for patch channel
    for part in range(n_parts):
        current_part_idx = parts_idx[part]
        n_current_part = len(current_part_idx)
        current_idx = []
        for i in range(n_current_part):
            current_start_idx = current_part_idx[i]
            current_idx.extend(range(current_start_idx * channel, current_start_idx * channel + channel))
        part_channel_idx.append(current_idx)


def get_cropped_face_cv(img, face_box3):
    face_box = face_box3.transpose()[:, :2]
    face_box_template = np.float32([[0, 0], [init_w, 0], [init_w, init_h]])
    M = cv2.getAffineTransform(face_box, face_box_template)
    M_inv = cv2.getAffineTransform(face_box_template, face_box)
    warped_face = cv2.warpAffine(img, M, (init_w, init_h))
    return warped_face, M_inv, M


def get_normalized_face_cv(img, pts, pose_idx):
    src_points = get_warp3points(pts)
    dst_points = img_big_template_parts[pose_idx]                  # warp to front template
    M = cv2.getAffineTransform(src_points, dst_points)
    M_inv = cv2.getAffineTransform(dst_points, src_points)
    pts_re = np.reshape(pts, (n_points, 1, 2))
    warped_pts = cv2.transform(pts_re, M)
    warped_pts = np.reshape(warped_pts, (n_points, 2))
    warped_face = cv2.warpAffine(img, M, (init_w, init_h))
    return warped_face, M_inv, M, warped_pts


def img_to_net_input(img, h, w):                                                                        # data transform
    """transform image to network input"""
    if img[0].shape == (h, w, 3):
        transposed_img = img.transpose(0, 3, 1, 2)
    elif img[0].shape == (h, w):
        transposed_img = img
    else:
        print('Error: input image shape error')
        return 0
    return transposed_img


def get_pose_by_pts(pts):                   # get pose index from points
    normalized_error_min = 999999999
    pose_idx = 0
    for i in range(0, 3):
        warp_mat_t, warped_pts = get_warp_mat_t_pts2template(pts, i)
        normalized_error = np.sum(np.sum(np.linalg.norm(warped_pts - mean_shapes_2d[i], axis=1)))
        if normalized_error < normalized_error_min:
            normalized_error_min = normalized_error
            pose_idx = i
    return pose_idx


def get_warp3points(pts):
    points3 = np.zeros((3, 2), np.float32)
    points3[0, :] = np.average(pts[PART_IDX_LEYE, :], axis=0)
    points3[1, :] = np.average(pts[PART_IDX_REYE, :], axis=0)
    points3[2, :] = np.average(pts[PART_IDX_MOUTH, :], axis=0)
    return points3


def get_warp_mat_t(src, dst):
    s = np.hstack((src, np.ones((3, 1), np.float32)))
    d = np.hstack((dst, np.ones((3, 1), np.float32)))
    return np.dot(np.linalg.inv(s), d)


def get_warped_pts(pts, warp_mat_t):
    shapes = pts.shape
    src_pts = np.hstack((pts, np.ones((shapes[0], 1), np.float32)))
    des_pts = np.dot(src_pts, warp_mat_t)
    return des_pts[:, 0:2]


def get_warp_mat_t_pts2template(pts, pose_idx):
    template_points = mean_shapes_2d[pose_idx]
    src_parts = get_warp3points(pts)
    dst_parts = get_warp3points(template_points)
    warp_mat_t = get_warp_mat_t(src_parts, dst_parts)
    src_1 = np.hstack((pts, np.ones((n_points, 1), np.float32)))
    warped_pts = np.dot(src_1, warp_mat_t)
    return warp_mat_t, warped_pts[:, 0:2]


def get_warp_mat_t_template2pts(pts, pose_idx):
    template_points = mean_shapes_2d[pose_idx]
    src_parts = get_warp3points(template_points)
    dst_parts = get_warp3points(pts)
    warp_mat_t = get_warp_mat_t(src_parts, dst_parts)
    return warp_mat_t


def template_pts2parameter(template_pts, pose_idx):
    shape = template_pts - mean_shapes_2d[pose_idx]
    shape_re = shape.reshape((n_points * 2, 1))
    coeff = np.dot(pinv_parameters[pose_idx], shape_re)
    coeff_normalized = np.divide(coeff, parameters_s[pose_idx])
    return coeff_normalized


def parameter2template_pts(coeff, pose_idx):
    coeff_denormalized = np.multiply(coeff, parameters_s[pose_idx])
    pts = np.dot(parameters[pose_idx], coeff_denormalized)
    pts = pts.reshape((n_points, 2))
    pts += mean_shapes_2d[pose_idx]
    return pts


def pts2parameter(pts, pose_idx):
    warp_mat_t2template, warped_pts = get_warp_mat_t_pts2template(pts, pose_idx)
    coeff = template_pts2parameter(warped_pts, pose_idx)
    return coeff


def parameter2pts(coeff, warp_mat_t, pose_idx):
    warped_pts = parameter2template_pts(coeff, pose_idx)
    pts = get_warped_pts(warped_pts, warp_mat_t)
    return pts


def get_mesh_warped_img(img, pts, pose_idx):
    """Piecewise affine warp to template mesh"""
    src_points = mean_shapes_mesh_ex[pose_idx]                              # mesh template size: 0 ~ mesh_h, 0 ~ mesh_w
    dst_points = np.zeros((n_points_ex, 2), np.float32)                     # current points with outside box
    dst_points[0:n_points, :] = pts                                         # current points
    warp_mat_t_template2pts = get_warp_mat_t_template2pts(pts, pose_idx)    # template -> pts warp matrix
    outside_pts = get_warped_pts(mean_shape_ex_outside, warp_mat_t_template2pts)     # warp outside points
    dst_points[n_points:, :] = outside_pts                                  # current points with outside box

    tform = PiecewiseAffineTransform()                                      # piecewise affine transform
    tform.estimate(src_points, dst_points)
    img_mesh_warped = warp(img, tform, output_shape=(mesh_h, mesh_w))

    # # draw
    # plt.figure(2)
    # plt.gcf().clear()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.scatter(src_points[:, 0], src_points[:, 1], c='r')
    # plt.subplot(1, 2, 2)
    # plt.imshow(img)
    # plt.scatter(dst_points[:, 0], dst_points[:, 1], c='r')
    # plt.draw()
    # plt.pause(0.001)
    # k = 1
    # k = k + 1

    return img_mesh_warped


def get_warped_error_img(img, pts, pose_idx):
    warped_img = get_mesh_warped_img(img, pts, pose_idx)
    warped_error_img = ((warped_img * 255).astype(np.int) - warped_mean_appearances[pose_idx]) / 2 + 127
    return warped_error_img.astype(np.uint8)


def get_patches_serial_input(img, pts, pose_idx):
    patches = np.zeros((patch_size, patch_size, channel * n_points), np.uint8)
    src_points = get_warp3points(pts)
    dst_points = img_template_parts[pose_idx]
    M = cv2.getAffineTransform(src_points, dst_points)
    M_inv = cv2.getAffineTransform(dst_points, src_points)
    pts_re = np.reshape(pts, (n_points, 1, 2))
    warped_pts = cv2.transform(pts_re, M)
    warped_pts = np.reshape(warped_pts, (n_points, 2))

    for i in range(n_points):
        M2 = M + [[0, 0, patch_center_x - warped_pts[i, 0]], [0, 0, patch_center_y - warped_pts[i, 1]]]
        patch = cv2.warpAffine(img, M2, (patch_size, patch_size), flags=cv2.INTER_NEAREST)
        patches[:, :, i * channel:i * channel + channel] = patch

    # # draw
    # plt.figure(101)
    # plt.gcf().clear()
    # for i in range(n_points):
    #     plt.subplot(7, 10, i+1)
    #     plt.imshow(patches[:, :, i*channel:i*channel+channel])
    # plt.draw()
    # plt.pause(0.001)
    # z=0

    return patches.transpose((2, 0, 1)), warped_pts, M_inv                                              # order: c, h, w

