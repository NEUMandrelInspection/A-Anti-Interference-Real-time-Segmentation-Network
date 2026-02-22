import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from lion_pytorch import Lion
from atorch.optimizers import AGD
from net_module import REN, MCAM, RDN, DSDH, GSPG, CFAM
from load_pred_unit import (count_parameters, load_and_prepare_data, perform_inference)
from tra_unit import unified_train_model
from loss_unit import (SegmentationClassificationLoss, DiceLoss, IoULoss)
import datetime


# 1. Configuration & Hyperparameters
epochs = 30  # Total training epochs
lion_epochs = 0  # Epochs using the Lion optimizer
batch_size = 16  # Batch size
mask_output_only = 1  # 1: Output mask images only, 0: Output triplet images
use_pretrained_model = 0  # 1: Load pre-trained model, 0: Train from scratch
only_val_output = 1  # 1: Infer on validation set only, 0: Infer on both training and validation sets
inference_compute_only = 0  # 1: Compute metrics only during inference (no saving), 0: Save predicted mask images
save_false_positives_only = 0  # 1: Save False Positive (FP) samples only during inference, 0: Save all
draw_bbox_on_inference = 0  # 1: Enable bounding box and confidence display, 0: Disable
use_optimizer = 2  # 2: Use SGD optimizer, 1: Use AdamW optimizer, 0: Use AGD optimizer
use_data_augmentation = 1  # 1: Enable training data augmentation (flip/rotate), 0: Disable
compare_initial_iou = 0  # 1: Compute IoU after loading weights, save only if new IoU is higher; 0: Use default minimum Loss saving strategy

lion_lr = 0.0003   # Lion optimizer learning rate
agd_lr = 0.0005  # AGD optimizer learning rate
agd_delta = 0.001  # AGD optimizer delta parameter
adamw_lr = 0.0001    # AdamW optimizer learning rate
sgd_lr = 0.001           # SGD learning rate
sgd_momentum = 0.9       # SGD momentum
sgd_weight_decay = 0.0005  # SGD weight decay

net_name = r"AIRSegNet" # Model name
source_data_dir = r"01Dataset" # Dataset path

# --- Dynamic Weight Configuration ---
dynamic_weight_enabled = False  # Enable dynamic weight for Dice loss (target samples only)
labeled_dice_weight_start = 0.1     # Initial weight
labeled_dice_weight_end = 0.3     # Final weight


# 2. Model Definition
class AIRSegNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_channels=4):
        super(AIRSegNet, self).__init__()

        # Channel configuration
        self.c1, self.c2, self.c3, self.c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        total_prototypes = self.c2 + self.c3

        # Instantiate the six modules from the architecture
        self.REN = REN(in_channels, self.c1, self.c2, self.c3, self.c4)
        self.MCAM = MCAM(self.c1, self.c2, self.c3)
        self.RDN = RDN(self.c1, self.c2, self.c3, self.c4, mcam_in_channels = self.c1 + self.c2 + self.c3)
        self.DSDH = DSDH(self.c1, self.c2, self.c3, self.c4, total_prototypes)
        self.GSPG = GSPG(gate_in_channels=3 * self.c1)
        self.CFAM = CFAM(in_channels, img_feat_channels=8, mask_channels=1)

        # Final output layer
        self.final_bn = nn.BatchNorm2d(num_classes)
        self.final_conv = nn.Conv2d(1, num_classes, kernel_size=1)


    def forward(self, x):
        H_orig, W_orig = x.shape[2], x.shape[3]
        current_x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True) if (H_orig != 512) else x

        # 1. Residual Encoding Network (REN)
        e1, e2, e3, e4 = self.REN(current_x)

        # 2. Multi-scale Cross-layer Attention Module (MCAM)
        mcam_out2, mcam_out3 = self.MCAM(e1, e2, e3)

        # 3. Residual Decoding Network (RDN)
        d2, d3, d4, mcam3_red, mcam2_red = self.RDN(e4, e2, mcam_out2, mcam_out3)

        # 4. Dual-path Semantic Decoupling Decision Head (DSDH)
        gate_logits, initial_mask, feat_map_global = self.DSDH(d2, d3, d4, mcam2_red, mcam3_red)

        # 5. Global Structure-aware Generation (GSPG)
        global_attention_map = self.GSPG(feat_map_global)
        filtered_mask = initial_mask * global_attention_map

        # 6. Cross-domain Feature Alignment Module (CFAM)
        aligned_mask, refined_logits = self.CFAM(current_x, filtered_mask, gate_logits)

        # Final processing
        out = self.final_conv(self.final_bn(aligned_mask))
        final_output_logits = F.interpolate(out, size=(H_orig, W_orig), mode='bilinear', align_corners=True)

        return torch.sigmoid(final_output_logits), torch.sigmoid(refined_logits)


# 3. Training Function
def initialize_model_and_optimizer(config, device):
    print("\n--- 2. Model Initialization and Pre-trained Weights Loading ---")
    model = AIRSegNet(in_channels=1).to(device)
    print(f"Total model parameters: {count_parameters(model)}\n")

    # --- Initialize total loss function including segmentation and classification ---
    seg_loss_config = {'bce_weight': 0.6, 'dice_weight': 0.2, 'iou_weight': 0.2}
    # --- labeled_dice_weight is used as default or when dynamic weight is disabled ---
    criterion = SegmentationClassificationLoss(
        seg_loss_config=seg_loss_config,
        class_loss_weight=config['class_loss_weight'],
        labeled_dice_weight=config['labeled_dice_weight_start'] if config['dynamic_weight_enabled'] else 0.2
    )

    # Loss functions for monitoring
    monitor_criterions = {'bce': nn.BCELoss(), 'dice': DiceLoss(), 'iou': IoULoss(), 'mse': nn.MSELoss()}

    # --- Determine initial optimizer based on configuration ---
    if config['lion_epochs'] > 0:
        print(f"--- Initial Optimizer: Lion (lr={config['lion_lr']}) ---")
        optimizer = Lion(model.parameters(), lr=config['lion_lr'], weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=config['lion_epochs'], eta_min=config['lion_lr'] / 1)

    else:
        if use_optimizer == 0:
            print(f"--- Initial Optimizer: AGD (lr={config['agd_lr']}) ---")
        optimizer = AGD(model.parameters(), lr=config['agd_lr'], delta=config['agd_delta'])
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config['agd_lr'] / 1)

    if config['use_pretrained_model'] == 1 and os.path.exists(config['default_best_model_path']):
        state_dict = torch.load(config['default_best_model_path'], map_location=torch.device('cpu'), weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"Pre-trained model weights loaded: {config['default_best_model_path']}")
    elif config['use_pretrained_model'] == 1:
        print(f"Warning: Pre-trained model not found at {config['default_best_model_path']}. Training from scratch.")

    return model, criterion, optimizer, scheduler, monitor_criterions


# 4. Main Execution Module
def main():

    train_data_path = os.path.join(source_data_dir, 'target', 'tra_target')
    val_data_path = os.path.join(source_data_dir, 'target', 'val_target')
    train_normal_path = os.path.join(source_data_dir, 'normal', 'tra_normal')
    val_normal_path = os.path.join(source_data_dir, 'normal', 'val_normal')
    output_dir = r"02Output"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Configuration Dictionary ---
    config = {
        'source_data_dir': source_data_dir,
        'train_data_path': train_data_path,
        'val_data_path': val_data_path,
        'train_normal_path': train_normal_path,
        'val_normal_path': val_normal_path,
        'output_dir': output_dir,
        'agd_lr': agd_lr,
        'agd_delta': agd_delta,
        'lion_lr': lion_lr,
        'lion_epochs': lion_epochs,
        'adamw_lr': adamw_lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': 512,
        'net_name': f"{net_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
        'default_best_model_path': os.path.join(output_dir, net_name + '.pth'),
        'use_pretrained_model': use_pretrained_model,
        'class_loss_weight': 0.25,
        'labeled_dice_weight': 0.2,
        'include_normal_in_training': 1,
        'use_optimizer': use_optimizer,
        'compare_initial_iou': compare_initial_iou,
        'inference_compute_only': inference_compute_only,
        'save_false_positives_only': save_false_positives_only,
        'draw_bbox_on_inference': draw_bbox_on_inference,
        'dynamic_weight_enabled': dynamic_weight_enabled,
        'only_val_output': only_val_output,
        'mask_output_only': mask_output_only,
        'use_data_augmentation': use_data_augmentation,
        'labeled_dice_weight_start': labeled_dice_weight_start,
        'labeled_dice_weight_end': labeled_dice_weight_end,
    }

    # 1. Data Loading
    train_dataloader, val_dataloader, train_data_infos, val_data_infos = load_and_prepare_data(config)

    # 2. Model, Optimizer, and Loss Function Initialization
    model, criterion, optimizer, scheduler, monitor_criterions = initialize_model_and_optimizer(config, device)

    # 3. Training
    unified_train_model(model, train_dataloader, val_dataloader, criterion,monitor_criterions,
                        optimizer, scheduler, device, config)

    # 4. Pre-inference Check
    if config['epochs'] == 0 and config['use_pretrained_model'] == 0:
        print("Inference skipped: epochs is 0 and no pre-trained model loaded. Please set epochs > 0 or use_pretrained_model = 1.")
        return

    # 5. Perform Inference on Training Set
    if only_val_output == 0:
        perform_inference(model, train_data_infos, config['output_dir'], device, set_name='training_set',
                          batch_size=2 * config['batch_size'], monitor_criterions=monitor_criterions,
                          compute_only=config['inference_compute_only'],
                          save_false_positives_only=config['save_false_positives_only'],
                          draw_bbox_on_inference=config['draw_bbox_on_inference'], mask_output_only=mask_output_only)

    # 6. Perform Inference on Validation Set
    perform_inference(model, val_data_infos, config['output_dir'], device, set_name='validation_set',
                      batch_size=2 * config['batch_size'], monitor_criterions=monitor_criterions,
                      compute_only=config['inference_compute_only'],
                      save_false_positives_only=config['save_false_positives_only'],
                      draw_bbox_on_inference=config['draw_bbox_on_inference'], mask_output_only=mask_output_only)


if __name__ == "__main__":
    main()