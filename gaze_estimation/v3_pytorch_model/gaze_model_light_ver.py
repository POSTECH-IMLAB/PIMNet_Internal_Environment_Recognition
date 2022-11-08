import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F

class Estimator(nn.Module):
    def __init__(self, use_mtcnn=False):
        super(Estimator, self).__init__()

        self.global_estimator = Global_Estimator()
        self.use_mtcnn = use_mtcnn
        

    def forward(self, input_x, flds=None):
        
        output = self.global_estimator(input_x)
        return output




# ------------------------------------- GLOBAL ---------------------
class Global_Estimator(nn.Module):
    def __init__(self):
        super(Global_Estimator, self).__init__()

        input_dim = 1
        cnum = 16


        self.lrelu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(3, 2)
        

        # 120 x 180
        self.conv1 = conv2d_block(input_dim, 20, 7, 2, 0)
        self.norm_1 = nn.InstanceNorm2d(20)

        # 60 x 90
        self.conv2 = conv2d_block(20, 32, 5, 2, 1)
        self.norm_2 = nn.InstanceNorm2d(32)

        # 30 x 45
        self.conv3 = conv2d_block(32, 30, 3, 1, 0)
        self.norm_3 = nn.InstanceNorm2d(30)
        
        self.conv4 = conv2d_block(30, 20, 3, 1, 0)
        #self.conv4_att = dis_conv(80, 1, 3, 1, 1)
        self.norm_4 = nn.InstanceNorm2d(20)

        self.conv5 = conv2d_block(20, 50, 3, 1, 0)
        self.norm_5 = nn.InstanceNorm2d(50)

        self.fc1 = nn.Linear((20 * 7 * 6) + (50 * 7 * 6), 2000)
        self.fc2 = nn.Linear(2000, 6)
    

    def forward(self, x):
        #print("ORIG -" + str(x.size()))
        x = F.pad(x, (53, 53, 63, 63)) # [left, right, top, bot]
        x = self.lrelu(self.conv1(x))
        x = self.norm_1(x)
        x = self.pool3(x)

        x = F.pad(x, (25, 25, 30, 30)) # [left, right, top, bot]
        x = self.lrelu(self.conv2(x))
        x = self.norm_2(x)
        x = self.pool(x)

        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x = self.lrelu(self.conv3(x))
        x = self.norm_3(x)
        x = self.pool(x)

        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x = self.lrelu(self.conv4(x))
        x = self.norm_4(x)
        x = self.pool(x)
        x_41 = x.view(x.size()[0], -1)

        x = F.pad(x, (1, 1, 1, 1)) 
        x = self.lrelu(self.conv5(x))
        x = self.norm_5(x)
        x_51 = x.view(x.size()[0], -1)

        # concat 41 & 51
        x = self.fc1(torch.cat((x_41, x_51), dim=1))
        x = self.fc2(x)
        
        return x







# ------ conv blocks -----------

class conv2d_block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2, padding=0, stddev=0.02):
        super(conv2d_block, self).__init__()
    
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=padding)
    def forward(self, x):
        return self.conv(x)

