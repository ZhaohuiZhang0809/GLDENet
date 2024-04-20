class LRRCB(nn.Module):
    def __init__(self, ch_in, ch_out, res=True):
        super(LRRCB, self).__init__()

        self.large_conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, dilation=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.small_conv = LocalFFN(ch_in)

        self.res = res

        self.downsample = nn.Sequential()
        if ch_in != ch_out:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        identity = self.small_conv(x)
        # identity = x
        x = self.large_conv(x)

        if self.res == True:
            x = x + self.downsample(identity)

        return x


class LocalFFN(nn.Module):
    r"""Local Feed-Forward Network"""
    def __init__(self, dim, stride=1, padding=2, dilation=2, expand_ratio=4):
        super(LocalFFN, self).__init__()
        hidden_dim = dim * expand_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding, dilation, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),   
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.DWConv(x)
        x = self.conv2(x)

        return x
