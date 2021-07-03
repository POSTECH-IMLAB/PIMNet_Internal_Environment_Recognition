# make input of initialization network
import glob
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
import fa_util as fu
import fa_util_train as fut
import h5py
import cv2


list_file_name = 'K:/VGG_list/vgg_list_all_000.txt'
output_prefix = 'K:/VGG_hdf5/init/VGG_wild_000'
# img_folders = ['../sample_data']
img_folders = ['D:/DB/FaceAlignment/HS_distribution/front', 'D:/DB/FaceAlignment/HS_distribution/left', 'D:/DB/FaceAlignment/HS_distribution/right']
# output_prefix = 'M:/HS_hdf5/wild/HS_wild'
jittering_size = 1                                                                     # should be changed 32*128
chunk_size = 1536                                                                        # should be changed


def get_part_pts(gt_pts, warp_mat_inv):
    part_centers = fu.get_part_centers(gt_pts)
    part_gt_pts = np.hstack((part_centers, np.ones((len(part_centers), 1), np.float32)))
    part_gt_pts_t = np.transpose(part_gt_pts)
    part_pts3 = np.dot(warp_mat_inv, part_gt_pts_t)
    return np.transpose(part_pts3)[:, 0:2]


def main():
    # files = fut.make_file_list_by_folder(img_folders, ['png', 'jpg'])   # get image file list by folder
    files = fut.make_file_list_by_text(list_file_name)                # get image file list by text file
    n_samples = len(files)
    random.seed(1234)                                                   # set random seed
    random.shuffle(files)                                               # random shuffle
    image_data_sets = fut.make_chunk_set(files, chunk_size)             # get image file chunk set
    print('Total number of samples: ' + str(n_samples))

    cnt_all = 0                                                         # cnt for sample images
    for i in range(len(image_data_sets)):
        current_num_img_files = len(image_data_sets[i])
        img_all = np.zeros((current_num_img_files * jittering_size, fu.init_h, fu.init_w, fu.channel), np.uint8)
        pts_all = np.zeros((current_num_img_files * jittering_size, fu.n_points, 2), np.float32)
        # part_all = np.zeros((current_num_img_files * jittering_size, fu.n_parts, 2), np.float32)

        # generate data
        cnt = 0                                                         # cnt for total samples with jittering
        for x in image_data_sets[i]:
            current_img_set = np.zeros((jittering_size, fu.init_h, fu.init_w, fu.channel), np.uint8)
            current_pts_set = np.zeros((jittering_size, fu.n_points, 2), np.float32)
            # current_part_set = np.zeros((jittering_size, fu.n_parts, 2), np.float32)
            cnt_all = cnt_all + 1
            print(str(datetime.now()) + ' (' + str(cnt_all) + '/' + str(n_samples) + ') ' + x)
            img, gt_pts = fut.load_img_pts(x)
            face_box3 = fut.get_bounding_box3_square_with_margin(gt_pts)
            for k in range(jittering_size):
                if k == 0:
                    face_box3_jittered = face_box3
                else:
                    face_box3_jittered = fut.get_jittered_bounding_box3(face_box3)


                img_face, M_inv, M = fu.get_cropped_face_cv(img, face_box3_jittered)



                pts = cv2.transform(gt_pts.reshape((fu.n_points, 1, 2)), M)

                pts = pts.reshape((fu.n_points, 2))

                # warp_mat_inv = np.linalg.inv(warp_mat)
                # pts = fu.get_warped_pts(gt_pts, warp_mat_inv.transpose())
                # part_pts = get_part_pts(gt_pts, warp_mat_inv)

                current_img_set[k, :, :, :] = img_face
                current_pts_set[k, :, :] = pts
                # current_part_set[k, :, :] = part_pts

                # # draw
                # plt.figure(1)
                # plt.gcf().clear()
                # plt.imshow(img_face)
                # plt.scatter(pts[:, 0], pts[:, 1], c='b')
                # # plt.scatter(part_pts[:, 0], part_pts[:, 1], c='r')
                # plt.draw()
                # plt.pause(0.001)
                # z = 0
            img_all[cnt:cnt + jittering_size, :, :, :] = current_img_set
            pts_all[cnt:cnt + jittering_size, :, :] = current_pts_set
            # part_all[cnt:cnt + jittering_size, :, :] = current_part_set
            cnt = cnt + jittering_size
        img_all = img_all.transpose((0, 3, 1, 2))  # order: sample, c, m, n
        pts_all[:, :, 0] = pts_all[:, :, 0] / fu.init_w  # normalize to 0~1
        pts_all[:, :, 1] = pts_all[:, :, 1] / fu.init_h  # normalize to 0~1
        # part_all[:, :, 0] = part_all[:, :, 0] / fu.init_w  # normalize to 0~1
        # part_all[:, :, 1] = part_all[:, :, 1] / fu.init_h  # normalize to 0~1

        suffle_idx = np.random.permutation(current_num_img_files * jittering_size)  # suffle
        img_all = img_all[suffle_idx, :, :, :]
        pts_all = pts_all[suffle_idx, :, :]
        # part_all = part_all[suffle_idx, :, :]

        current_output_path = "%s_%03d.h5" % (output_prefix, i)
        hf = h5py.File(current_output_path, 'w')
        input_face_img_name = "img"
        warped_img_set = hf.create_dataset(input_face_img_name, data=img_all)
        input_pts_name = "pts"
        pts_set = hf.create_dataset(input_pts_name, data=pts_all)
        # input_part_name = "part"
        # part_set = hf.create_dataset(input_part_name, data=part_all)
        hf.close()


if __name__ == "__main__":
    main()
