import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResBlock, self).__init__()

        padding1 = (kernel_size - 1) // 2
        if kernel_size == 1:
            padding1 = 0

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        padding2 = padding1

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding2,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 2. Shortcut Connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            # Key fix: Force shortcut connection to use 1x1 conv, padding=0
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0,
                          bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        # Copy input as the start of the shortcut connection
        identity = x

        # Forward propagation of the main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut output (potentially after 1x1 conv) to the main path output
        out += self.shortcut(identity)  # Dimensions should match here

        # Activation after addition
        out = self.relu(out)

        return out


class DoubleResBlock(nn.Module):
    expansion = ResBlock.expansion

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(DoubleResBlock, self).__init__()

        self.out_channels_final = out_channels * self.expansion

        # Shortcut connection definition
        if stride != 1 or in_channels != self.out_channels_final:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels_final,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels_final)
            )
        else:
            self.shortcut = nn.Identity()

        # First ResBlock
        self.resblock1 = ResBlock(in_channels, out_channels, stride=stride, kernel_size=kernel_size)

        # Second ResBlock
        self.resblock2 = ResBlock(self.out_channels_final, out_channels, stride=1, kernel_size=kernel_size)

    def forward(self, x):

        shortcut_out = self.shortcut(x)

        out1 = self.resblock1(x)
        out_final = self.resblock2(out1)

        # Overall residual connection
        out = out_final + shortcut_out

        return out


class IntegratedAttentionModule(nn.Module):

    def __init__(self, in_channels, c2_channels_factor=0.5, reduction=4):
        super(IntegratedAttentionModule, self).__init__()
        self.in_channels = in_channels

        # Calculate channels for the multi-scale fusion part (C2)
        c2_channels = int(in_channels * c2_channels_factor)
        if c2_channels < 1:
            c2_channels = 1

        # Initial 1x1 conv
        self.conv1x1_initial = nn.Conv2d(in_channels, c2_channels, kernel_size=1, bias=False)

        # --- Initialize upper part layers (Multi-Scale Extraction) ---
        # Multi-Scale Fusion
        self.msf_conv3x3 = nn.Conv2d(c2_channels, c2_channels, kernel_size=3, padding=1, bias=False)
        self.msf_conv5x5 = nn.Conv2d(c2_channels, c2_channels, kernel_size=5, padding=2, bias=False)
        self.msf_conv7x7 = nn.Conv2d(c2_channels, c2_channels, kernel_size=7, padding=3, bias=False)

        # Spatial Aggregation layers
        self.spatial_agg_conv = nn.Conv2d(c2_channels, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_agg_sigmoid = nn.Sigmoid()

        # Final 1x1 conv to restore channels
        self.conv1x1_final = nn.Conv2d(c2_channels, in_channels, kernel_size=1, bias=False)

        # --- Initialize lower part layers (Channel Aggregation) ---
        reduced_dim = in_channels // reduction
        if reduced_dim < 1:
            reduced_dim = 1

        # Channel Aggregation layers
        self.channel_agg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_agg_fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # Save original input for the final residual connection
        identity = x

        # =================== Upper Path ===================
        # 1. Initial 1x1 conv
        x_main = self.conv1x1_initial(x)

        # 2. Multi-Scale Fusion
        x3 = self.msf_conv3x3(x_main)
        x5 = self.msf_conv5x5(x_main)
        x7 = self.msf_conv7x7(x_main)
        x_msf = x3 + x5 + x7  # Sum the results

        # 3. Spatial Aggregation (implemented directly here)
        #   a. 7x7 conv
        spatial_attention_conv = self.spatial_agg_conv(x_msf)
        #   b. Sigmoid to generate spatial attention map
        spatial_attention_map = self.spatial_agg_sigmoid(spatial_attention_conv)

        # 4. Apply Spatial Attention
        # Multiply the spatial attention map with the output of conv1x1_initial
        x_spatial_weighted = x_main * spatial_attention_map

        # 5. Final 1x1 conv
        x_upper_path = self.conv1x1_final(x_spatial_weighted)

        # =================== Lower Path ===================
        # 1. Channel Aggregation (implemented directly here)
        #   a. Average pooling
        channel_weights_pooled = self.channel_agg_pool(x)
        #   b. Two 1x1 convs
        channel_weights = self.channel_agg_fc(channel_weights_pooled)

        # =================== Combine and Output ===================
        # 1. Apply Channel Attention
        # Multiply the upper path result with the channel weights
        x_channel_weighted = x_upper_path * channel_weights

        # 2. Residual connection
        # Add the weighted result to the original input
        out = x_channel_weighted + identity

        return out


class GroupedChannelAttention(nn.Module):  # Abbreviated as GCA, formerly SemanticEnhancer
    """
    This is a feature-map to feature-map module used to enhance semantic information.
    """

    def __init__(self, in_channels, num_groups=4, fc_ratio=0.25):
        super().__init__()

        self.num_groups = num_groups
        if in_channels % num_groups != 0:
            # Adjust number of groups if not divisible
            for i in range(num_groups, 0, -1):
                if in_channels % i == 0:
                    self.num_groups = i
                    break
            else:
                self.num_groups = 1

        self.group_channels = in_channels // self.num_groups

        mid_channels = max(1, int(in_channels * fc_ratio))
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=False)
        )

        self.weight_generator = nn.Conv2d(in_channels, in_channels, 1, 1, 0,
                                          groups=self.num_groups, bias=True)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Split input feature map x into n groups (X_i)
        # (b, c, h, w) -> (b, n, c_group, h, w)
        x_groups = x.view(b, self.num_groups, self.group_channels, h, w)

        # 2. Extract global semantic info Xc
        # (b, c, h, w) -> (b, c, 1, 1)
        global_ctx = self.global_context(x)

        # 3. Generate n sets of dynamic weights w_i^c
        # (b, c, 1, 1) -> (b, c, 1, 1)
        weights = self.weight_generator(global_ctx)

        # 4. Weight normalization (Softmax)
        # (b, c, 1, 1) -> (b, n, c_group, 1, 1)
        weights = weights.view(b, self.num_groups, self.group_channels, 1, 1)
        weights = self.softmax(weights)

        # 5. Feature enhancement (X_i * w_i^c)
        enhanced_groups = x_groups * weights

        # 6. Re-concatenate
        # (b, n, c_group, h, w) -> (b, c, h, w)
        out = enhanced_groups.view(b, c, h, w)

        return out


class DualStreamGateHead(nn.Module):   # Softmax Channel Attention Module (Latest)
    """
    Dual-stream gate head, generating feature maps for channel weights and global attention respectively.
    1. Receives semantic features from different decoder levels g2, g3, g4.
    2. After fusing features, processes them through two parallel GroupedChannelAttention streams.
    3. Each stream outputs a dedicated feature map.
    4. Uses conv and pooling layers to extract info from both feature maps, jointly generating classification logits.
    """

    def __init__(self, in_channels, base_channels, se_groups=4, fc_ratio=0.25):
        super().__init__()
        # in_channels = 3 * base_channels (from fused features g2, g3, g4)
        self.in_channels = in_channels

        # Stream 1: For channel weights generating prototype masks (Channel Attention)
        self.stream_channel = GroupedChannelAttention(in_channels, num_groups=se_groups, fc_ratio=fc_ratio)

        # Stream 2: For generating global spatial attention map (Global Attention)
        self.stream_global = GroupedChannelAttention(in_channels, num_groups=se_groups, fc_ratio=fc_ratio)

        # Feature extractor replacing GAP, used to extract info from both streams and generate classification logits
        # Input is concatenated feature maps from two streams, channels = 2 * in_channels
        self.feature_extractor_for_logits = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Finally use avg pooling to compress into a vector
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels, 1)
        )


    def forward(self, g2, g3, g4):
        # g2 (B, C1, /8), g3 (B, C1, /16), g4 (B, C1, /32)

        # 1. Align sizes to g3 and fuse (same as before)
        g4_up = F.interpolate(g4, size=g3.shape[2:], mode='bilinear', align_corners=False)
        g2_down = F.interpolate(g2, size=g3.shape[2:], mode='bilinear', align_corners=False)
        fused_semantic_feat = torch.cat([g4_up, g3, g2_down], dim=1)  # (B, 3*C1, H/16, W/16)

        # 2. Process through two parallel streams
        # Add residual connection F(x) + x
        feat_map_channel = self.stream_channel(fused_semantic_feat) + fused_semantic_feat
        feat_map_global = self.stream_global(fused_semantic_feat) + fused_semantic_feat

        # 3. Extract features and generate classification logits
        # Concatenate both feature maps
        combined_feat_map = torch.cat([feat_map_channel, feat_map_global], dim=1)  # (B, 2 * 3*C1, H/16, W/16)

        # Pass through feature extractor
        extracted_vector = self.feature_extractor_for_logits(combined_feat_map)  # (B, 3*C1, 1, 1)
        extracted_vector = extracted_vector.squeeze(-1).squeeze(-1)  # (B, 3*C1)

        # Generate logits
        gate_logits = self.classifier(extracted_vector)  # (B, 1)

        # Return logits and the two dedicated feature maps
        return gate_logits, feat_map_channel, feat_map_global


class PrototypeMaskRefinementHead(nn.Module):
    """
    Generates prototype masks of specified channels based on input feature maps.
    Includes an upsampling layer and two convolutional layers.
    """
    def __init__(self, in_channels, num_prototypes, upsample_stride=1):
        super().__init__()
        # Add an upsampling layer to increase resolution before generating masks for finer details
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=upsample_stride,padding=0,stride=upsample_stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        # Two conv layers to refine features and generate prototypes
        self.proto_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, num_prototypes, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.upsample(x)
        prototypes = self.proto_conv(feat)
        return prototypes


class MaskWeightingHead(nn.Module):
    """
    Performs weighted summation on multi-channel prototype masks to obtain the final single-channel segmentation mask.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prototypes, weights):
        """
        Args:
            prototypes (torch.Tensor): B, num_prototypes, H, W
            weights (torch.Tensor): B, num_prototypes
        """
        # Expand weights from (B, P) to (B, P, 1, 1) for broadcast multiplication
        # P = num_prototypes
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # Weighting: (B, P, H, W) * (B, P, 1, 1) -> (B, P, H, W)
        weighted_prototypes = prototypes * weights

        # Summation: Sum along the prototype channel dimension to get a single-channel mask
        # (B, P, H, W) -> (B, 1, H, W)
        final_mask = torch.sum(weighted_prototypes, dim=1, keepdim=True)

        return final_mask


class ConvAttentionGenerator(nn.Module):
    """
    A purely convolutional attention generator, replacing TransformerAttentionGenerator.

    It achieves this through the following steps:
    1. A multi-scale dilated convolution branch to capture contextual info at different ranges.
    2. A parallel residual connection to preserve original information.
    3. Finally generates a single-channel attention map.
    """

    def __init__(self, in_channels, target_size=128):
        """
        Initialize the module.
        Args:
            in_channels (int): Input feature map channels (e.g., 3*C1 in version 3.5).
            target_size (int): Target size of the final output attention map.
        """
        super().__init__()
        self.target_size = target_size

        # Define an intermediate channel size for multi-scale branches to avoid excessive channels
        mid_channels = in_channels // 4 if in_channels > 4 else in_channels

        # --- 1. Multi-scale Dilated Convolution Branch ---
        # 1.1 Reduce channels with 1x1 conv first
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 1.2 Parallel multi-scale dilated convolutions
        self.dilated_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.dilated_conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3, bias=False)
        self.dilated_conv5 = nn.Conv2d(mid_channels, mid_channels, kernel_size=11, padding=5, bias=False)

        # 1.3 BN and ReLU after each branch
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # --- 2. Residual Connection Branch ---
        # Unify channels using 1x1 conv
        self.skip_connection = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        # --- 3. Downsample-Upsample Structure ---
        # Downsampling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling layers (Transposed Convolution)
        # Upsample features after the first pooling
        self.upsample1 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        # Upsample features after the second pooling
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)  # Two consecutive upsamples
        )

        # --- 4. Generate Final Attention Map ---
        # After fusing all features, generate a single-channel map using 1x1 conv
        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Record original input size for final upsampling
        B, C, H_in, W_in = x.shape

        # --- Steps 1 & 2: Multi-scale Branch & Residual Connection ---
        # Multi-scale branch
        feat_reduced = self.reduce_conv(x)
        feat_d1 = self.dilated_conv1(feat_reduced)
        feat_d3 = self.dilated_conv3(feat_reduced)
        feat_d5 = self.dilated_conv5(feat_reduced)
        multi_scale_out = self.bn_relu(feat_d1 + feat_d3 + feat_d5)

        # Residual connection
        skip_out = self.skip_connection(x)

        # Fuse by addition
        fused_feat = multi_scale_out + skip_out

        # --- Step 3: Downsample-Upsample & Feature Fusion ---
        # First downsample
        p1 = self.pool1(fused_feat)
        # Second downsample
        p2 = self.pool2(p1)

        # Upsample the downsampled features to restore original fused_feat size
        up1 = self.upsample1(p1)
        up2 = self.upsample2(p2)

        # Core step: Add features from three scales together
        # 1. Original multi-scale result (fused_feat)
        # 2. Result after one pooling and one transposed conv (up1)
        # 3. Result after two poolings and two transposed convs (up2)
        final_fused_feat = fused_feat + up1 + up2

        # --- Step 4: Generate Final Attention Map ---
        # Upsample fused features to target size
        attention_map = F.interpolate(final_fused_feat, size=(self.target_size, self.target_size), mode='bilinear',
                                      align_corners=False)

        # Get final single-channel attention map through final_conv
        attention_map = self.final_conv(attention_map)

        return attention_map


class ImageDownsampler(nn.Module):
    """
    Downsamples the input image by 4x, extracting features combining intermediate layers and residual connections.
    Input: (B, 1, 512, 512)
    Output: (B, C, 128, 128)
    """

    def __init__(self, in_channels=1, base_out_channels=2):
        super(ImageDownsampler, self).__init__()
        self.c1_out = base_out_channels
        self.c2_out = base_out_channels * 4

        # First level downsampling: 512 -> 256
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, self.c1_out, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.c1_out),
            nn.ReLU(inplace=True)
        )

        # Second level downsampling: 256 -> 128
        self.down2 = nn.Sequential(
            nn.Conv2d(self.c1_out, self.c2_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(self.c2_out, self.c2_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.c2_out),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv for skip connection
        self.skip1_conv = nn.Conv2d(self.c1_out, self.c2_out, kernel_size=1, stride=2, bias=False)
        self.skip_orig_conv = nn.Conv2d(in_channels, self.c2_out, kernel_size=1, stride=4, bias=False)  # Direct 512->128


    def forward(self, x):
        # Residual connection of the original image
        skip_orig = self.skip_orig_conv(x)

        # First level downsampling and skip connection
        d1_out = self.down1(x)
        skip1 = self.skip1_conv(d1_out)

        # Second level downsampling
        d2_out = self.down2(d1_out)

        # Feature fusion
        out = d2_out + skip1 + skip_orig

        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class AFEM(nn.Module):  # Adaptive Feature Enhancement Module
    """
    Feature Enhancement Branch, a generic module for FEB-2, 3, 4 based on SFA paper.
    """

    def __init__(self, in_channels, out_channels, config):
        super(AFEM, self).__init__()
        self.config = config

        if config == 'FEB-2':
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif config == 'FEB-3':
            # Using DepthwiseSeparableConv to simulate DWConv
            self.dw_conv1 = DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 1, 1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.dw_conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 1, 1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.c_conv = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)  # Concat and 1x1 conv
            self.bn_c = nn.BatchNorm2d(out_channels)
        elif config == 'FEB-4':
            self.dw_conv1 = DepthwiseSeparableConv(in_channels, out_channels, 3, 1, 2, 2)  # Dilation 2
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.dw_conv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, 1, 5, 5)  # Dilation 5
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.c_conv = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)
            self.bn_c = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        if self.config == 'FEB-2':
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:  # FEB-3 and FEB-4
            f1 = self.relu(self.bn1(self.dw_conv1(x)))
            f2 = self.relu(self.bn2(self.dw_conv2(f1)))
            out_cat = torch.cat([f1, f2], dim=1)
            out = self.bn_c(self.c_conv(out_cat))

        out += identity
        return self.relu(out)


class CSAM(nn.Module):  # Channel-Spatial Alignment Module
    """
    Spatial and Channel Attention Module
    """

    def __init__(self, in_channels):
        super(CSAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = x * self.channel_attention(x)
        sa = ca * self.spatial_attention(ca)
        return sa


class AlignmentModule(nn.Module):
    """
    Alignment module, fusing image features and mask features (added residual connection)
    """

    def __init__(self, img_feat_channels, mask_channels, base_align_channels=16):
        super(AlignmentModule, self).__init__()
        in_channels = img_feat_channels + mask_channels

        self.initial_conv = nn.Conv2d(in_channels, base_align_channels, 1)

        # Multi-scale feature enhancement branches (Formerly FEB, now AFEM)
        self.feb2 = AFEM(base_align_channels, base_align_channels, 'FEB-2')
        self.feb3 = AFEM(base_align_channels, base_align_channels, 'FEB-3')
        self.feb4 = AFEM(base_align_channels, base_align_channels, 'FEB-4')

        # Concat and SCA
        self.sca_in_channels = base_align_channels * 3
        self.sca = CSAM(self.sca_in_channels)

        # --- Added: 1x1 conv for the first residual connection ---
        # Ensures channel dimensions match so SCA output can be added to the FEB fused features
        self.residual_conv1 = nn.Conv2d(self.sca_in_channels, self.sca_in_channels, 1)

        # Final processing
        self.final_conv = nn.Conv2d(self.sca_in_channels, 1, 1)  # Output a single channel mask

        # A simple head for logits, using GAP on the aligned features
        logit_head_inter_channels = self.sca_in_channels // 2  # Channels can be appropriately reduced

        self.logit_head = nn.Sequential(
            nn.Conv2d(self.sca_in_channels, logit_head_inter_channels, kernel_size=3, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(logit_head_inter_channels, 1)  # Note: Input channels need updating here
        )

    def forward(self, img_feat, mask_feat, gate_logits_in):
        # Concatenate image features and mask
        x = torch.cat([img_feat, mask_feat], dim=1)
        x = self.initial_conv(x)

        # Get features from all branches
        f2 = self.feb2(x)
        f3 = self.feb3(x)
        f4 = self.feb4(x)

        # Concatenate features from branches
        f_cat = torch.cat([f2, f3, f4], dim=1)

        # Apply Spatial-Channel Attention
        f_aligned_raw = self.sca(f_cat)

        # --- Residual Connection 1: Add SCA output and FEB fused features ---
        f_aligned = self.residual_conv1(f_aligned_raw) + f_cat

        # Generate final mask refinement
        mask_refinement = self.final_conv(f_aligned)

        # --- Residual Connection 2: Add refinement amount to the original input mask ---
        final_mask = mask_refinement + mask_feat

        # Refine logits
        refined_logits = self.logit_head(f_aligned) + gate_logits_in

        return final_mask, refined_logits