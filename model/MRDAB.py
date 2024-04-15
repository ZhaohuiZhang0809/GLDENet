class Dwt2d(nn.Module):
    def __init__(self):
        super(Dwt2d, self).__init__()
        self.requires_grad = False

    def dwt(self, x):
        with torch.no_grad():
            x = x.cpu()    ##
            # LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
            LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')

            LL = torch.tensor(LL).cuda()
            LH = torch.tensor(LH).cuda()
            HL = torch.tensor(HL).cuda()
            HH = torch.tensor(HH).cuda()

        return torch.cat((LL, LH, HL, HH), 1)

    def forward(self, x):
        out = self.dwt(x)
        return out
        

class MRDAB(nn.Module):
    r""" Multi-scale Receptive-field Downsampling Attention Block """
    def __init__(self, dim, ratio=4, scale=8):
        super(MRDAB, self).__init__()
        self.ratio = ratio

        self.pool = nn.MaxPool2d(2)
        # self.pool = nn.AdaptiveAvgPool2d((size // 2, size // 2))
        self.dwt = Dwt2d()

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//ratio, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim//ratio),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(dim, 1, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, stride=1, bias=False)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=False)
        # self.conv5 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, stride=1, bias=False)

        self.channel_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(dim, dim // scale, kernel_size=1),
            nn.LayerNorm([dim // scale, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // scale, dim, kernel_size=1)
        )

        self.conv1x1 = nn.Conv2d(dim * 6, dim, 3, padding=1, stride=1,bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
            p = self.pool(x)

            B, C, H, W = list(x.size())
            qkv0 = self.dwt(self.reduce(x))

            # Multi-scale Receptive field
            qkv1 = self.conv1(qkv0)
            qkv2 = self.conv2(qkv0)
            qkv3 = self.conv3(qkv0)
            qkv4 = self.conv4(qkv0)

            qkv = self.conv1x1(torch.cat([qkv0, qkv1, qkv2, qkv3, qkv4, p], dim=1))

            # channel att
            content_feature = self.conv(qkv).view(B, -1, W//2 * H//2).permute(0, 2, 1)  # HW x 1
            content_feature = self.softmax(content_feature)

            channel_feature = self.channel_conv(qkv).view(B, -1, W//2 * H//2)  # C x HW
            channel_pooling = torch.bmm(channel_feature, content_feature).view(B, -1, 1, 1)  # C x 1 x 1
            channel_weight = self.channel_trans_conv(channel_pooling)
            att_cha = qkv * channel_weight

            x = self.proj(att_cha)
            
            return x
