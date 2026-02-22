import torch
import torch.nn as nn
import torch.nn.functional as F
from net_unit import (DoubleResBlock, IntegratedAttentionModule, DualStreamGateHead,
                      PrototypeMaskRefinementHead, MaskWeightingHead, ConvAttentionGenerator,
                      ImageDownsampler, AlignmentModule)

# --- 1. REN (Residual Encoder Network)  ---
class REN(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(REN, self).__init__()
        self.encoder_block1 = DoubleResBlock(in_channels, c1, stride=2)
        self.encoder_block2 = DoubleResBlock(c1, c2, stride=2)
        self.encoder_block3 = DoubleResBlock(c2, c3, stride=2)
        self.encoder_block4 = DoubleResBlock(c3, c4, stride=2)

    def forward(self, x):
        e1 = self.encoder_block1(x)  # /2  256
        e2 = self.encoder_block2(e1)  # /4 128
        e3 = self.encoder_block3(e2)  # /8 64
        e4 = self.encoder_block4(e3)  # /16 32
        return e1, e2, e3, e4


# --- 2. MCAM (Multi-scale Cross-level Attention Module) ---
class MCAM(nn.Module):
    def __init__(self, c1, c2, c3):
        super(MCAM, self).__init__()
        self.mcam_in_channels = c1 + c2 + c3
        self.mcam_level2 = IntegratedAttentionModule(in_channels=self.mcam_in_channels)
        self.mcam_level3 = IntegratedAttentionModule(in_channels=self.mcam_in_channels)

    def forward(self, e1, e2, e3):
        e1_to_e3 = F.interpolate(e1, size=e3.shape[2:], mode='bilinear', align_corners=True)
        e2_to_e3 = F.interpolate(e2, size=e3.shape[2:], mode='bilinear', align_corners=True)
        fpn_concat_3 = torch.cat([e1_to_e3, e2_to_e3, e3], dim=1)
        mcam_out3 = self.mcam_level3(fpn_concat_3)

        e1_to_e2 = F.interpolate(e1, size=e2.shape[2:], mode='bilinear', align_corners=True)
        e3_to_e2 = F.interpolate(e3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        fpn_concat_2 = torch.cat([e1_to_e2, e2, e3_to_e2], dim=1)
        mcam_out2 = self.mcam_level2(fpn_concat_2)

        return mcam_out2, mcam_out3


# --- 3. RDN (Residual Decoder Network) ---
class RDN(nn.Module):
    def __init__(self, c1, c2, c3, c4, mcam_in_channels):
        super(RDN, self).__init__()
        self.decoder_4 = DoubleResBlock(c4, c4)
        self.upconv4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.mcam_c3_reduction = nn.Conv2d(mcam_in_channels, c3, kernel_size=1, bias=False)
        self.decoder_3 = DoubleResBlock(c3 + c3, c3)
        self.upconv3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.mcam_c2_reduction = nn.Conv2d(mcam_in_channels, c2, kernel_size=1, bias=False)
        self.decoder_2 = DoubleResBlock(c2 + c2 + c2, c2)


    def forward(self, e4, e2, mcam_out2, mcam_out3):
        d4 = self.decoder_4(e4)

        mcam3_reduced = self.mcam_c3_reduction(mcam_out3)
        d3_in = torch.cat([self.upconv4(d4), mcam3_reduced], dim=1)
        d3 = self.decoder_3(d3_in)

        mcam2_reduced = self.mcam_c2_reduction(mcam_out2)
        d2_in = torch.cat([self.upconv3(d3), e2, mcam2_reduced], dim=1)
        d2 = self.decoder_2(d2_in)

        return d2, d3, d4, mcam3_reduced, mcam2_reduced


# --- 4. DSDH (Dual-path Semantic Decoupling Head) ---
class DSDH(nn.Module):
    def __init__(self, c1, c2, c3, c4, total_prototypes):
        super(DSDH, self).__init__()
        self.gate_in_channels = 3 * c1
        self.total_prototypes = total_prototypes

        self.gate_d4_reduce = nn.Sequential(nn.Conv2d(c4, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True))
        self.gate_d3_reduce = nn.Sequential(nn.Conv2d(c3, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True))
        self.gate_d2_reduce = nn.Sequential(nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True))
        self.mcam_g3_reduce = nn.Sequential(nn.Conv2d(c3, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True))
        self.mcam_g2_reduce = nn.Sequential(nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(True))
        self.gate_head = DualStreamGateHead(in_channels=self.gate_in_channels, base_channels=c1)

        # Basis Factory
        self.proto_gen_d3 = PrototypeMaskRefinementHead(c3, c3, upsample_stride=2)
        self.proto_gen_d2 = PrototypeMaskRefinementHead(c2, c2, upsample_stride=1)

        # Weighting Head
        self.SemanticPrototypeGatingModule = nn.Sequential(
            nn.Conv2d(self.gate_in_channels, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(True),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c1, total_prototypes), nn.Softmax(dim=1)
        )
        self.mask_weighting = MaskWeightingHead()


    def forward(self, d2, d3, d4, mcam2_red, mcam3_red):

        g4 = self.gate_d4_reduce(d4)
        g3 = self.gate_d3_reduce(d3) + self.mcam_g3_reduce(mcam3_red)
        g2 = self.gate_d2_reduce(d2) + self.mcam_g2_reduce(mcam2_red)

        gate_logits, feat_map_channel, feat_map_global = self.gate_head(g2, g3, g4)
        proto_3 = self.proto_gen_d3(d3)
        proto_2 = self.proto_gen_d2(d2)
        all_prototypes = torch.cat([proto_2, proto_3], dim=1)

        mask_weights = self.SemanticPrototypeGatingModule(feat_map_channel)
        initial_mask = self.mask_weighting(all_prototypes, mask_weights)

        return gate_logits, initial_mask, feat_map_global


# --- 5. GSPG (Global Structural Perception Generator) ---
class GSPG(nn.Module):
    def __init__(self, gate_in_channels):
        super(GSPG, self).__init__()
        self.global_attention_gen = ConvAttentionGenerator(in_channels=gate_in_channels, target_size=128)

    def forward(self, feat_map_global):
        return self.global_attention_gen(feat_map_global)


# --- 6. CFAM (Cross-domain Feature Alignment Module) ---
class CFAM(nn.Module):
    def __init__(self, in_channels, img_feat_channels=8, mask_channels=1):
        super(CFAM, self).__init__()
        # (Feature Sampler)
        self.image_downsampler = ImageDownsampler(in_channels=in_channels, base_out_channels=2)
        self.alignment_module = AlignmentModule(img_feat_channels=img_feat_channels, mask_channels=mask_channels)

    def forward(self, current_x, initial_mask, gate_logits):
        img_features_low_res = self.image_downsampler(current_x)
        aligned_mask, refined_logits = self.alignment_module(img_features_low_res, initial_mask, gate_logits)
        return aligned_mask, refined_logits


