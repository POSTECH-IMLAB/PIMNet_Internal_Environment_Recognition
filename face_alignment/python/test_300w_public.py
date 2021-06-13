import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import face_alignment as fa
import fa_util_train as fut

img_folders = ['N:\DB\FaceAlignment\\300W_public_test\lfpw', 'N:\DB\FaceAlignment\\300W_public_test\helen', 'N:\DB\FaceAlignment\\300W_public_test\ibug']
# img_folders = ['D:\DB\FaceAlignment\\300W_public_test\ibug']
# img_folders = ['../sample_data2']
img_extension = ['png', 'jpg']
output_folder = '../result'
max_iter = 21
# max_iter = 1

def main():
    # fa.fa_init([[1], [0, 0], [0, 0]])
    fa.fa_init([[1], [1], [0, 0], [1, 0]])
    files = []
    current_pts = np.zeros((max_iter+1, 68, 2), np.float32)
    error_IOD = np.zeros((max_iter+1, 1), np.float32)
    error_BOX = np.zeros((max_iter+1, 1), np.float32)
    for folder in img_folders:
        for ext in img_extension:
            current_files = glob.glob(folder + '/*.' + ext)
            files.extend(current_files)

    cnt = 0
    n_samples = len(files)
    for x in files:
        cnt += 1

        img, gt_pts = fut.load_img_pts(x)
        face_box3 = fut.get_bounding_box3_square_with_margin(gt_pts)
        # current_pts[0, :, :], current_pts[1, :, :], current_pts[2, :, :], pose_idx = fa.face_alignment_detection(img, face_box3, -1)
        # current_pts[0, :, :], _, current_pts[1, :, :], pose_idx = fa.face_alignment_detection(img, face_box3, -1)
        current_pts[0, :, :], current_pts[1, :, :], current_pts[2, :, :], pose_idx = fa.face_alignment_detection(img, face_box3, -1)

        for i in range(2, max_iter):
            current_pts[i+1, :, :], pose_idx = fa.face_alignment_detection_step(img, current_pts[i, :, :], pose_idx)

        print(str(datetime.now()) + ' ' + str(cnt) + '/' + str(n_samples) + ' ' + x)
        for i in range(0, max_iter):
            output_path = fut.get_output_path(x, output_folder, 'pt%d' % i)
            fut.save_pts(output_path, current_pts[i, :, :])
            error_IOD[i, 0], error_BOX[i, 0] = fut.measurement(gt_pts, current_pts[i, :, :])
            print('Error%d :' % i + str(error_IOD[i, 0]))

        # draw
        draw_idx = [0, 1, 2, 3, 4]
        # draw_idx = [0]
        plt.figure(1)
        plt.gcf().clear()
        draw_cnt = 1
        for i in draw_idx:
            plt.subplot(1, len(draw_idx), draw_cnt)
            plt.imshow(img)
            plt.scatter(current_pts[i, :, 0], current_pts[i, :, 1], s=3, c='r')
            draw_cnt += 1
        plt.draw()
        plt.pause(0.001)
        z = 1


if __name__ == "__main__":
    main()
