import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, downsample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_chanels = in_channels

        if downsample:
            self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=1, padding=1)
            self.batch_norm1 = nn.BatchNorm3d(in_channels)
            self.batch_norm2 = nn.BatchNorm3d(in_channels*2)
        else:
            self.conv1 = nn.Conv3d(in_channels=(in_channels + in_channels//2), out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
            self.batch_norm1 = nn.BatchNorm3d(in_channels//2)
            self.batch_norm2 = nn.BatchNorm3d(in_channels//2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        return x
    
    
class UNet3D(nn.Module):
    def __init__(self, in_channels = 1, out_channels=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_dim2 = 32

        self.down_conv1 = nn.Conv3d(in_channels=in_channels, out_channels=self.in_dim2, kernel_size=3, stride=1, padding=1)
        self.down_conv2 = nn.Conv3d(in_channels=self.in_dim2, out_channels=self.in_dim2*2, kernel_size=3, stride=1, padding=1)

        self.batch_norm_down1 = nn.BatchNorm3d(self.in_dim2)
        self.batch_norm_down2 = nn.BatchNorm3d(self.in_dim2*2)

        self.relu_down_1 = nn.ReLU()
        self.relu_down_2 = nn.ReLU()

        self.max_pool_down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.max_pool_down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.max_pool_down3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.double_conv_down1 = DoubleConv(in_channels=self.in_dim2*2) #out is in_dim2*4
        self.double_conv_down2 = DoubleConv(in_channels=self.in_dim2*4) #out is in_dim2*8
        self.double_conv_down3 = DoubleConv(in_channels=self.in_dim2*8) #out is in_dim2*16

        self.conv_up1 = nn.ConvTranspose3d(in_channels=self.in_dim2*16, out_channels=self.in_dim2*16, kernel_size=2, stride=2)
        self.conv_up2 = nn.ConvTranspose3d(in_channels=self.in_dim2*8, out_channels=self.in_dim2*8, kernel_size=2, stride=2)
        self.conv_up3 = nn.ConvTranspose3d(in_channels=self.in_dim2*4, out_channels=self.in_dim2*4, kernel_size=2, stride=2)

        self.double_conv_up1 = DoubleConv(in_channels=self.in_dim2*16, downsample=False) #out is in_dim*8
        self.double_conv_up2 = DoubleConv(in_channels=self.in_dim2*8, downsample=False) #out is in_dim*4
        self.double_conv_up3 = DoubleConv(in_channels=self.in_dim2*4, downsample=False) #out is in_dim*2

        self.final_conv = nn.Conv3d(in_channels=self.in_dim2*2, out_channels=out_channels, stride=1, kernel_size=1, padding=0)

    def forward(self, x):

        x = self.down_conv1(x)
        x = self.batch_norm_down1(x)
        x = self.relu_down_1(x)
        x = self.down_conv2(x)
        x = self.batch_norm_down2(x)
        x = self.relu_down_2(x)

        y = self.max_pool_down1(x)
        y = self.double_conv_down1(y)

        z = self.max_pool_down2(y)
        z = self.double_conv_down2(z)

        v = self.max_pool_down3(z)
        v = self.double_conv_down3(v)

        v = self.conv_up1(v)

        z_concat = torch.cat([z, v], dim=1)
        z_final = self.double_conv_up1(z_concat)
        z_final = self.conv_up2(z_final)

        y_concat = torch.cat([y, z_final], dim=1)
        y_final = self.double_conv_up2(y_concat)
        y_final = self.conv_up3(y_final)

        x_concat = torch.cat([x, y_final], dim=1)
        x_final = self.double_conv_up3(x_concat)
        x_final = self.final_conv(x_final)

        return x_final






        


