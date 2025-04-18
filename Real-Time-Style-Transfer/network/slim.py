from torch import nn


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding=kernel_size//2)

    def forward(self, x):
        out = self.conv2d(x)
        return out


# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, scale_factor=None):
        super(UpsampleConvLayer, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = None
        if scale_factor is not None:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


# Residual Layer
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU6(inplace=True)

        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))

        out = self.in2(self.conv2(out))

        out = out + residual
        out = self.relu(out)
        return out


# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()

        # nonlineraity
        self.relu = nn.ReLU6(inplace=True)

        # encoding layers
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.in1_e = nn.InstanceNorm2d(16, affine=True)

        self.conv2 = ConvLayer(16, 16, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(16, affine=True)

        self.conv3 = ConvLayer(16, 16, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(16, affine=True)

        # residual layers
        self.res = ResidualBlock(16)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(16, 16, kernel_size=3, stride=1, scale_factor=2)
        self.in3_d = nn.InstanceNorm2d(16, affine=True)

        self.deconv2 = UpsampleConvLayer(16, 16, kernel_size=3, stride=1, scale_factor=2)
        self.in2_d = nn.InstanceNorm2d(16, affine=True)

        self.deconv1 = UpsampleConvLayer(16, 3, kernel_size=3, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))

        y = self.relu(self.in2_e(self.conv2(y)))

        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))

        y = self.deconv1(y)

        return y
