class MAAB(nn.Module):
    r"""multi-scale attention aggregation block"""
    def __init__(self, in_channels_1:int , in_channels_2:int, ratio=8):
        super(MAAB, self).__init__()
        self.query_conv = nn.Conv2d(in_channels_1, in_channels_1 // ratio, kernel_size=3, padding=1, bias=False)       # in_ch1 < in_ch2
        self.key_conv = nn.Conv2d(in_channels_2, in_channels_1 // ratio, kernel_size=3, padding=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels_2, in_channels_1, kernel_size=3, padding=1, bias=False)

        self.conv = nn.Conv2d(in_channels_1, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.channel_conv = nn.Conv2d(in_channels_1, in_channels_1, kernel_size=3, padding=1, bias=False)

        self.channel_trans_conv = nn.Sequential(
            nn.Conv2d(in_channels_1, in_channels_1//ratio, kernel_size=1),
            nn.LayerNorm([in_channels_1//ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_1//ratio, in_channels_1, kernel_size=1)
        )


    def forward(self, x1, x2):
        B, C1, W1, H1 = x1.size()
        B, C2, W2, H2 = x2.size()
        query = self.query_conv(x1).view(B, -1, W1 * H1)
        key = self.key_conv(x2).view(B, -1, W2 * H2).permute(0, 2, 1)
        value = self.value_conv(x2).view(B, -1, W2 * H2)

        content = torch.bmm(key, query)
        content = self.softmax(content)
        att_con = torch.bmm(value, content).view(B, -1, W1, H1) + x1

        content_feature = self.conv(x1).view(B, -1, W1 * H1).permute(0, 2, 1)       # HW x 1
        content_feature = self.softmax(content_feature)
        channel_feature = self.channel_conv(x1).view(B, -1, W1 * H1)                # C x HW
        channel_pooling = torch.bmm(channel_feature, content_feature).view(B, -1, 1, 1)       # C x 1 x 1
        channel_weight = self.channel_trans_conv(channel_pooling)
        att_cha = x1 * channel_weight

        out = att_con + att_cha

        return out
