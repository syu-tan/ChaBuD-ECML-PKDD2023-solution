import torch
from torch import nn
import torch.nn.functional as F
import timm

class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']

class TimmUnet(AbstractModel):
    def __init__(self, 
                    in_chans=24, 
                    pretrained=True,
                    channels_last=False, 
                    out_chans=1,
                    encoder="resnet34",
                    decoder_filters=[48, 96, 176, 256],
                    last_upsample=48,
                    **kwargs):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        self.channels_last = channels_last
        backbone = timm.create_model(backbone_arch, features_only=True, in_chans=in_chans, pretrained=pretrained,)
        self.filters = [f["num_chs"] for f in backbone.feature_info]

        self.decoder_filters = decoder_filters
        self.last_upsample_filters = last_upsample

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])
        self._mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, out_chans)

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        # self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.encoder(x)
        x = enc_results[-1]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        mask = self._mask(x).contiguous(memory_format=torch.contiguous_format)
        return mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])



class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class UnetDecoderLastConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.layer(x)
    
class SiameseFuseUnet(AbstractModel):
    def __init__(self, 
                    in_chans=24, 
                    pretrained=True,
                    channels_last=False, 
                    out_chans=1,
                    encoder="resnet34",
                    decoder_filters=[48, 96, 176, 256],
                    last_upsample=48,
                    **kwargs):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        self.channels_last = channels_last
        backbone = timm.create_model(backbone_arch, features_only=True, in_chans=int(in_chans/2), pretrained=pretrained,)
        self.filters = [f["num_chs"] for f in backbone.feature_info]

        self.decoder_filters = decoder_filters
        self.last_upsample_filters = last_upsample

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2]*2 + f*2, f*2) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])
        self._mask = UnetDecoderLastConv(self.decoder_filters[0]*2, self.last_upsample_filters, out_chans)

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        # self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            
        x1 = x[:, :12, :, :] # B, C:12, H, W; Psot
        x2 = x[:, 12:, :, :] # B, C:12, H, W; Pre
            
        enc_results1 = self.encoder(x1)
        enc_results2 = self.encoder(x2)
        
        x = torch.cat([enc_results1[-1], enc_results2[-1]], dim=1)
        
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            fuse = torch.cat([x, enc_results1[rev_idx - 1], enc_results2[rev_idx - 1]], dim=1)
            x = bottleneck(x, fuse)

        mask = self._mask(x).contiguous(memory_format=torch.contiguous_format)
        return mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels*2, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    
    
if __name__ == '__main__':
    
    # encoder_params = {
    # "resnet34": {
    #     "decoder_filters": [96, 176, 192],
    #     "last_upsample": 32
    #     }
    # }

    # default_decoder_filters = [96, 176, 256]
    # default_last = 48
    
    # shape = (2, 24, 512, 512)
    # inputs = torch.zeros(shape)
    
    # model = TimmUnet(encoder='hrnet_w30', in_chans=24, pretrained=True, channels_last=False,)
    # outputs = model(inputs)
    # print(model.name)
    # print(outputs.shape)
    
    encoder_params = {
    "resnet34": {
        "decoder_filters": [96, 176, 192],
        "last_upsample": 32
        }
    }

    default_decoder_filters = [96, 176, 256]
    default_last = 48
    
    shape = (2, 24, 512, 512)
    inputs = torch.zeros(shape)
    
    model = SiameseFuseUnet(encoder='hrnet_w30', in_chans=24, pretrained=True, channels_last=False,)
    outputs = model(inputs)
    print(model.name)
    print(outputs.shape)