import caffe
import numpy as np
import matplotlib.pyplot as plt
import fa_util as fu
import cv2

nets = [[], [], [], [[[], [], []], []]]
set_load_nets = []   # network load table
# deploy_path = [['../models/ZF_deploy.prototxt'], ['../models/RES_ln_global_front_deploy.prototxt', '../models/RES_ln_global_left_deploy.prototxt', '../models/RES_ln_global_right_deploy.prototxt'], []]
deploy_path = [['../models/HS_wild_pts68_deploy_rfcn.prototxt'],
               ['../models/HS_wild2_front_deploy.prototxt', '../models/HS_wild2_left_deploy.prototxt', '../models/HS_wild2_right_deploy.prototxt'],
               ['../models/RES_ln_global_front_deploy.prototxt', '../models/RES_ln_global_left_deploy.prototxt', '../models/RES_ln_global_right_deploy.prototxt'],
               ['../models/RES_local_deploy_front_%s.prototxt', '../models/RES_local_deploy_left_%s.prototxt', '../models/RES_local_deploy_right_%s.prototxt']]

# weight_path = [['../caffemodels/HS_wild_pts68_iter_830000.caffemodel'],['../caffemodels\RES_ln_global_front_iter_400000.caffemodel', '../caffemodels\RES_ln_global_left_iter_400000.caffemodel', '../caffemodels\RES_ln_global_right_iter_400000.caffemodel'], []]
# weight_path = [['../caffemodels/HS_wild_pts68_iter_830000.caffemodel'],['../caffemodels\RES_L2_global_front_iter_400000.caffemodel', '../caffemodels\RES_ln_global_left_iter_400000.caffemodel', '../caffemodels\RES_ln_global_right_iter_400000.caffemodel'], []]
# weight_path = [['../caffemodels/FA_ZF_baseline_iter_150000.caffemodel'],['../caffemodels\RES_ln_global_front_iter_400000.caffemodel', '../caffemodels\RES_ln_global_left_iter_400000.caffemodel', '../caffemodels\RES_ln_global_right_iter_400000.caffemodel'], []]
# weight_path = [['N:/caffemodels/RES_ln_wild_pts68_iter_1000000.caffemodel'],
#                ['N:/caffemodels\RES_ln_global_front_iter_500000.caffemodel', 'N:/caffemodels\RES_ln_global_left_iter_500000.caffemodel', 'N:/caffemodels\RES_ln_global_right_iter_500000.caffemodel'],
#                ['N:/caffemodels\RES_local_front_%s_iter_250000.caffemodel', 'N:/caffemodels\RES_local_left_%s_iter_250000.caffemodel', 'N:/caffemodels\RES_local_right_%s_iter_250000.caffemodel']]
weight_path = [['N:/caffemodels/HS_wild_pts68_iter_600000.caffemodel'],
               ['N:/caffemodels\HS_wild2_front_iter_10000.caffemodel', 'N:/caffemodels\HS_wild2_left_iter_10000.caffemodel', 'N:/caffemodels\HS_wild2_right_iter_10000.caffemodel'],
               ['K:/hyunsung/caffemodels/RES_ln_global_front_iter_760000.caffemodel', 'K:/hyunsung/caffemodels/RES_ln_global_left_iter_760000.caffemodel', 'K:/hyunsung/caffemodels/RES_ln_global_right_iter_760000.caffemodel'],
               ['N:/caffemodels\RES_local_front_%s_iter_250000.caffemodel', 'N:/caffemodels\RES_local_left_%s_iter_250000.caffemodel', 'N:/caffemodels\RES_local_right_%s_iter_250000.caffemodel']]


def fa_init(net_load_table):
    global nets
    global set_load_nets
    set_load_nets = net_load_table
    # caffe.set_mode_gpu()
    # caffe.set_device(0)
    caffe.set_mode_cpu()
    fu.fu_init()

    # load networks
    if set_load_nets[0][0] == 1:                                                        # wild network
        nets[0].append(caffe.Net(deploy_path[0][0], weight_path[0][0], caffe.TEST))
    if set_load_nets[1][0] == 1:                                                        # wild2 network
        for pose in range(fu.n_pose):
            deploy_path_txt = deploy_path[1][pose]
            weight_path_txt = weight_path[1][pose]
            nets[1].append(caffe.Net(deploy_path_txt, weight_path_txt, caffe.TEST))
    if set_load_nets[2][0] == 1:                                                        # global network
        for pose in range(fu.n_pose):
            deploy_path_txt = deploy_path[2][pose]
            weight_path_txt = weight_path[2][pose]
            nets[2].append(caffe.Net(deploy_path_txt, weight_path_txt, caffe.TEST))
    if set_load_nets[3][0] == 1:                                                        # local network
        for pose in range(fu.n_pose):
            for part in range(fu.n_parts):
                deploy_path_txt = deploy_path[3][pose] % fu.part_txt[part]
                weight_path_txt = weight_path[3][pose] % fu.part_txt[part]
                nets[3][0][pose].append(caffe.Net(deploy_path_txt, weight_path_txt, caffe.TEST))


def face_alignment_detection(img, face_box, pose_idx):
    pts0 = wild_pts68_network(img, face_box)

    if pose_idx == -1:
        new_pose = fu.get_pose_by_pts(pts0)
    else:
        new_pose = pose_idx

    if set_load_nets[1][0] == 1:
        pts1 = wild2_network(img, pts0, new_pose)
        if set_load_nets[3][0] == 1:
            pts2 = local_network(img, pts1, new_pose)
        else:
            pts2 = np.zeros((68, 2), np.float32)
    elif set_load_nets[2][0] == 1:
        pts1 = global_network(img, pts0, new_pose)
        if set_load_nets[3][0] == 1:
            pts2 = local_network(img, pts1, new_pose)
        else:
            pts2 = np.zeros((68, 2), np.float32)
    elif set_load_nets[3][0] == 1:
        pts1 = np.zeros((68, 2), np.float32)
        pts2 = local_network(img, pts1, new_pose)
    else:
        pts1 = np.zeros((68, 2), np.float32)
        pts2 = np.zeros((68, 2), np.float32)

    # # draw
    # plt.figure(201)
    # plt.gcf().clear()
    # plt.subplot(1, 2, 1)
    # plt.imshow(warped_img)
    # plt.subplot(1, 2, 2)
    # plt.imshow((warped_error_img+1)/2)
    # plt.draw()
    # plt.pause(0.001)
    # z = 1

    return pts0, pts1, pts2, new_pose


def face_alignment_detection_step(img, pts, pose_idx):
    if set_load_nets[2][0] == 1:
        pts1 = global_network(img, pts, pose_idx)
        if set_load_nets[3][0] == 1:
            pts1 = local_network(img, pts1, pose_idx)
    elif set_load_nets[3][0] == 1:
        pts1 = local_network(img, pts, pose_idx)

    # new_pose = fu.get_pose_by_pts(pts)
    new_pose = pose_idx

    return pts1, new_pose


def wild_pts68_network(img, face_box):
    # input data
    img_face, M_inv, M = fu.get_cropped_face_cv(img, face_box)
    input_data = np.zeros((1, fu.init_h, fu.init_w, fu.channel), np.float32)
    input_data[0, :, :, :] = img_face
    input_data = fu.img_to_net_input(input_data, fu.init_h, fu.init_w)

    # network forwarding
    out = nets[0][0].forward(img=input_data)
    output = out[nets[0][0].outputs[0]]
    pts = output[0, :]
    wild_pts = np.reshape(pts, (fu.n_points, 2))
    wild_pts[:, 0] *= fu.init_w
    wild_pts[:, 1] *= fu.init_h

    pts_re = np.reshape(wild_pts, (fu.n_points, 1, 2))
    pts_result = cv2.transform(pts_re, M_inv)
    pts_out = np.reshape(pts_result, (fu.n_points, 2))

    # # draw
    # plt.figure(21)
    # plt.gcf().clear()
    # plt.imshow(img_face)
    # plt.scatter(pts_out[:, 0], pts_out[:, 1])
    # plt.draw()
    # plt.pause(0.001)
    # z=0

    return pts_out


def wild2_network(img, pts_in, pose_idx):
    # input data
    img_face, M_inv, M, predicted_pts = fu.get_normalized_face_cv(img, pts_in, pose_idx)
    input_data = np.zeros((1, fu.init_h, fu.init_w, fu.channel), np.float32)
    input_data[0, :, :, :] = img_face
    input_data = fu.img_to_net_input(input_data, fu.init_h, fu.init_w)

    # network forwarding
    out = nets[1][pose_idx].forward(img=input_data)
    output = out[nets[1][pose_idx].outputs[0]]
    pts = output[0, :]
    wild_pts = np.reshape(pts, (fu.n_points, 2))
    wild_pts[:, 0] *= fu.init_w
    wild_pts[:, 1] *= fu.init_h

    pts_re = np.reshape(wild_pts, (fu.n_points, 1, 2))
    pts_result = cv2.transform(pts_re, M_inv)
    pts_out = np.reshape(pts_result, (fu.n_points, 2))

    # # draw
    # plt.figure(21)
    # plt.gcf().clear()
    # plt.imshow(img_face)
    # plt.scatter(wild_pts[:, 0], wild_pts[:, 1], c='r')
    # plt.scatter(predicted_pts[:, 0], predicted_pts[:, 1], c='b')
    # plt.draw()
    # plt.pause(0.001)
    # z=0

    return pts_out


def global_network(img, pts_in, pose_idx):
    warped_error_img = fu.get_warped_error_img(img, pts_in, pose_idx)
    warped_error_img_input = np.zeros((1, fu.mesh_h, fu.mesh_w, fu.channel), np.float32)
    warped_error_img_input[0, :, :, :] = warped_error_img
    warped_error_img_input = warped_error_img_input.transpose((0, 3, 1, 2))

    warp_mat_t_pts2template, warped_pts = fu.get_warp_mat_t_pts2template(pts_in, pose_idx)
    current_coeff = fu.template_pts2parameter(warped_pts, pose_idx)
    out = nets[2][0].forward(warped_img=warped_error_img_input)
    delta_p = out[nets[2][pose_idx].outputs[0]]              # network forwarding
    delta_p /= fu.parameter_scale
    delta_p = np.reshape(delta_p, (fu.shape_dim, 1))
    coeff_new = current_coeff + delta_p
    warp_mat_t = fu.get_warp_mat_t_template2pts(pts_in, pose_idx)
    pts_reconstructed = fu.parameter2pts(coeff_new, warp_mat_t, pose_idx)
    return pts_reconstructed


def local_network(img, pts_in, pose_idx):
    # input data
    patches, warped_pts, M_inv = fu.get_patches_serial_input(img, pts_in, pose_idx)
    input_data = np.zeros((1, fu.channel * fu.n_points, fu.patch_size, fu.patch_size), np.float32)
    input_data[0, :, :, :] = patches
    pts_move = np.zeros((fu.n_points, 2), np.float32)

    for part in range(fu.n_parts):
        current_part_channel_idx = fu.part_channel_idx[part]
        current_n_part = fu.n_part_points[part]
        out = nets[3][0][pose_idx][part].forward(patch=input_data[:, current_part_channel_idx, :, :])                 # network forwarding
        output = out[nets[3][0][pose_idx][part].outputs[0]]
        moving_vector = output[0, :]
        moving_vector = np.reshape(moving_vector, (current_n_part, 2))
        pts_move[fu.parts_idx[part], :] = moving_vector

        # #draw
        # plt.figure(31)
        # plt.gcf().clear()
        # for i in range(current_n_part):
        #     plt.subplot(1, current_n_part, i + 1)
        #     img = input_data[0, current_part_channel_idx[i*fu.channel:i*fu.channel + fu.channel], :, :]
        #     img = img.transpose((1, 2, 0)).astype(np.uint8)
        #     plt.imshow(img)
        #
        # plt.draw()
        # plt.pause(0.01)
        # z=0

    # reconstruction
    temp = pts_move / fu.move_scale
    # temp = pts_move
    warped_pts += temp
    warped_pts = np.reshape(warped_pts, (fu.n_points, 1, 2))
    pts_out = cv2.transform(warped_pts, M_inv)
    pts_out = np.reshape(pts_out, (fu.n_points, 2))

    return pts_out

