class CFVB(nn.Module):
    r"""cross-layer fusion voting block"""
    def __init__(self, in_channels:int, ratio=8):
        super(CFVB, self).__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.channel_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1),
            nn.LayerNorm([in_channels//ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1)
        )

        # self.linear = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.voting_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x1, x2):
        B, C, W, H = x1.size()

        content_feature1 = self.conv(x1).view(B, -1, W * H).permute(0, 2, 1)  # HW x 1
        content_feature1 = self.softmax(content_feature1)
        channel_feature1 = self.channel_conv(x1).view(B, -1, W * H)  # C x HW
        channel_pooling1 = torch.bmm(channel_feature1, content_feature1).view(B, -1, 1, 1)  # C x 1 x 1
        channel_weight1 = self.channel_trans_conv(channel_pooling1)

        content_feature2 = self.conv(x2).view(B, -1, W * H).permute(0, 2, 1)  # HW x 1
        content_feature2 = self.softmax(content_feature2)
        channel_feature2 = self.channel_conv(x2).view(B, -1, W * H)  # C x HW
        channel_pooling2 = torch.bmm(channel_feature2, content_feature2).view(B, -1, 1, 1)  # C x 1 x 1
        channel_weight2 = self.channel_trans_conv(channel_pooling2)

        # channel_weight = self.linear(torch.cat([channel_weight1, channel_weight2], dim=1))
        channel_weight = channel_weight1 + channel_weight2

        # att_cha1 = x1 * channel_weight
        # att_cha2 = x2 * channel_weight

        # out = att_cha1 + att_cha2
        x2 = x2 * channel_weight
        # channel_weight = self.linear(torch.cat([channel_weight1, channel_weight2], dim=1))
        #
        # att_cha1 = x1 * channel_weight
        # att_cha2 = x2 * channel_weight
        #
        # att = torch.cat([att_cha1, att_cha2], dim=1)
        v1 = self.voting_gate(x2)
        # v2 = self.voting_gate(x1)

        # v = self.voting_gate(att)

        return v1
