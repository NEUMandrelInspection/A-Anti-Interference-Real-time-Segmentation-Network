import os
import numpy as np
import torch
from tqdm import tqdm
from loss_unit import calculate_dice, calculate_iou
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from atorch.optimizers import AGD

def unified_train_model(model, train_dataloader, val_dataloader, criterion, monitor_criterions, optimizer, scheduler,
                        device, config):
    print("\n--- 3. Start Model Training ---\n")
    if config['epochs'] <= 0:
        print("Training epochs set to 0. Skipping training.")
        return

    target_optimizer_type = config.get('use_optimizer', 2)
    compare_initial_iou = config.get('compare_initial_iou', 0)

    save_counter = 0
    best_val_weighted_loss = float('inf')
    best_val_iou = 0.0
    best_model_path = os.path.join(config['output_dir'], config['net_name'])

    if config['use_pretrained_model'] == 0 and os.path.exists(best_model_path):
        print(f"Note: 'use_pretrained_model' is 0. Training from scratch, ignoring existing weights at {best_model_path}.")

    if compare_initial_iou == 1:
        print("\n[Initial Check] compare_initial_iou=1 detected. Computing initial model IoU on validation set...")
        model.eval()
        init_val_iou_scores = []
        with torch.no_grad():
            for images, masks, categories, subtypes in tqdm(val_dataloader, desc="Initial Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                pred_masks, _ = outputs
                pred_mask_binary = (pred_masks > 0.5)
                pred_mask_numpy = pred_mask_binary.squeeze(1).cpu().numpy()
                true_mask_numpy = masks.squeeze(1).cpu().numpy()
                for i in range(pred_mask_numpy.shape[0]):
                    if np.any(true_mask_numpy[i]):
                        init_val_iou_scores.append(calculate_iou(pred_mask_numpy[i], true_mask_numpy[i]))

        initial_iou = np.mean(init_val_iou_scores) if init_val_iou_scores else 0.0
        best_val_iou = initial_iou
        print(f"[Initial Check] Initial Average IoU: {initial_iou:.4f}")
        print(f"[Strategy] New model will be saved only when validation IoU exceeds {best_val_iou:.4f}.\n")

    for epoch in range(config['epochs']):
        current_labeled_dice_weight = None
        if config.get('dynamic_weight_enabled', False):
            total_epochs = config['epochs']
            start_weight = config['labeled_dice_weight_start']
            end_weight = config['labeled_dice_weight_end']

            if total_epochs > 1:
                progress = epoch / (total_epochs - 1)
            else:
                progress = 1.0

            current_labeled_dice_weight = start_weight + (end_weight - start_weight) * progress
            if epoch == 0:
                print(f"Dynamic weight enabled: labeled_dice_weight will increase linearly from {start_weight} to {end_weight}.")
            print(f"Epoch {epoch + 1}: Current labeled_dice_weight = {current_labeled_dice_weight:.4f}")

        if config.get('lion_epochs', 0) >= 0 and epoch == config['lion_epochs']:
            remaining_epochs = config['epochs'] - epoch

            if target_optimizer_type == 1:
                print(f"--- Optimizer switched to AdamW at Epoch {epoch + 1} ---")
                print(f"--- AdamW learning rate set to: {config['adamw_lr']} ---\n")
                optimizer = optim.AdamW(model.parameters(), lr=config['adamw_lr'])
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=0.00005)

            elif target_optimizer_type == 0:
                print(f"--- Optimizer switched from Lion to AGD at Epoch {epoch + 1} ---")
                print(f"--- AGD learning rate set to: {config['agd_lr']}, delta: {config['agd_delta']} ---\n")
                optimizer = AGD(model.parameters(), lr=config['agd_lr'], delta=config['agd_delta'])
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=config['agd_lr'] / 10)

            elif target_optimizer_type == 2:
                sgd_lr = config.get('sgd_lr', 0.005)
                sgd_momentum = config.get('sgd_momentum', 0.9)
                sgd_weight_decay = config.get('sgd_weight_decay', 0.0005)

                print(f"--- Optimizer switched to **SGD** at Epoch {epoch + 1} ---")
                print(f"--- SGD parameters: LR={sgd_lr}, Momentum={sgd_momentum}, Weight Decay={sgd_weight_decay} ---")

                decay_params = []
                no_decay_params = []
                for name, param in model.named_parameters():
                    if not param.requires_grad: continue
                    if len(param.shape) == 1 or 'bias' in name.lower() or 'bn' in name.lower() or 'norm' in name.lower():
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                param_groups = [
                    {'params': decay_params, 'weight_decay': sgd_weight_decay},
                    {'params': no_decay_params, 'weight_decay': 0.0}
                ]
                optimizer = optim.SGD(param_groups, lr=sgd_lr, momentum=sgd_momentum)
                scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=sgd_lr / 100)
                print(f"--- SGD parameter grouping complete: {len(decay_params)} parameters apply decay, {len(no_decay_params)} parameters do not apply decay. ---\n")

        model.train()
        running_metrics = {k: 0.0 for k in ['bce', 'dice', 'iou', 'mse', 'weighted']}
        all_train_dice_scores = []
        all_train_iou_scores = []

        progress_bar_train = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} (Training)")
        for images, masks, categories, subtypes in progress_bar_train:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            pred_masks, _ = outputs

            target_class = torch.tensor([1.0 if cat == 'target' else 0.0 for cat in categories], device=device,
                                        dtype=torch.float)

            bce_weight_map = torch.ones_like(masks, device=device)

            total_weighted_loss = criterion(outputs, masks, target_class, bce_weight_map,
                                            dynamic_labeled_dice_weight=current_labeled_dice_weight)
            total_weighted_loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_metrics['weighted'] += total_weighted_loss.item()
                running_metrics['bce'] += monitor_criterions['bce'](pred_masks, masks).item()
                running_metrics['dice'] += monitor_criterions['dice'](pred_masks, masks).item()
                running_metrics['iou'] += monitor_criterions['iou'](pred_masks, masks).item()
                running_metrics['mse'] += monitor_criterions['mse'](pred_masks, masks).item()

                pred_mask_binary = (pred_masks > 0.5)
                pred_mask_numpy = pred_mask_binary.squeeze(1).cpu().numpy()
                true_mask_numpy = masks.squeeze(1).cpu().numpy()

                for i in range(pred_mask_numpy.shape[0]):
                    if np.any(true_mask_numpy[i]):
                        all_train_dice_scores.append(calculate_dice(pred_mask_numpy[i], true_mask_numpy[i]))
                        all_train_iou_scores.append(calculate_iou(pred_mask_numpy[i], true_mask_numpy[i]))

        scheduler.step()

        avg_train_weighted_loss = running_metrics['weighted'] / len(train_dataloader)
        avg_train_dice = np.mean(all_train_dice_scores) if all_train_dice_scores else 0.0
        avg_train_iou = np.mean(all_train_iou_scores) if all_train_iou_scores else 0.0

        print(f"\nEpoch {epoch + 1}/{config['epochs']}, Training Weighted Loss: {avg_train_weighted_loss:.6f}")
        print(f"Training Loss - BCE: {running_metrics['bce'] / len(train_dataloader):.6f}, "
              f"Dice: {running_metrics['dice'] / len(train_dataloader):.6f}, "
              f"IoU: {running_metrics['iou'] / len(train_dataloader):.6f}, "
              f"MSE: {running_metrics['mse'] / len(train_dataloader):.6f}")
        print(f"Training Average Dice (Labeled samples only): {avg_train_dice:.4f}, Training Average IoU (Labeled samples only): {avg_train_iou:.4f}\n")

        if val_dataloader and len(val_dataloader.dataset) > 0:
            model.eval()
            running_metrics_val = {k: 0.0 for k in ['bce', 'dice', 'iou', 'mse', 'weighted']}
            all_val_dice_scores = []
            all_val_iou_scores = []

            progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']} (Validation)")
            with torch.no_grad():
                for images, masks, categories, subtypes in progress_bar_val:
                    images, masks = images.to(device), masks.to(device)

                    outputs = model(images)
                    pred_masks, _ = outputs

                    target_class = torch.tensor([1.0 if cat == 'target' else 0.0 for cat in categories],
                                                device=device, dtype=torch.float)

                    bce_weight_map = torch.ones_like(masks, device=device)

                    total_weighted_loss = criterion(outputs, masks, target_class, bce_weight_map,
                                                    dynamic_labeled_dice_weight=current_labeled_dice_weight)
                    running_metrics_val['bce'] += monitor_criterions['bce'](pred_masks, masks).item()
                    running_metrics_val['dice'] += monitor_criterions['dice'](pred_masks, masks).item()
                    running_metrics_val['iou'] += monitor_criterions['iou'](pred_masks, masks).item()
                    running_metrics_val['mse'] += monitor_criterions['mse'](pred_masks, masks).item()
                    running_metrics_val['weighted'] += total_weighted_loss.item()

                    pred_mask_binary = (pred_masks > 0.5)
                    pred_mask_numpy = pred_mask_binary.squeeze(1).cpu().numpy()
                    true_mask_numpy = masks.squeeze(1).cpu().numpy()

                    for i in range(pred_mask_numpy.shape[0]):
                        if np.any(true_mask_numpy[i]):
                            all_val_dice_scores.append(calculate_dice(pred_mask_numpy[i], true_mask_numpy[i]))
                            all_val_iou_scores.append(calculate_iou(pred_mask_numpy[i], true_mask_numpy[i]))

            avg_val_dice = np.mean(all_val_dice_scores) if all_val_dice_scores else 0.0
            avg_val_iou = np.mean(all_val_iou_scores) if all_val_iou_scores else 0.0
            avg_val_weighted_loss = running_metrics_val['weighted'] / len(val_dataloader)

            print(f"\nEpoch {epoch + 1}/{config['epochs']}, Validation Weighted Loss: {avg_val_weighted_loss:.6f}")
            print(f"Validation Loss - BCE: {running_metrics_val['bce'] / len(val_dataloader):.6f}, "
                  f"Dice: {running_metrics_val['dice'] / len(val_dataloader):.6f}, "
                  f"IoU: {running_metrics_val['iou'] / len(val_dataloader):.6f}, "
                  f"MSE: {running_metrics_val['mse'] / len(val_dataloader):.6f}")
            print(f"Validation Average Dice (Labeled samples only): {avg_val_dice:.4f}, Validation Average IoU (Labeled samples only): {avg_val_iou:.4f}\n")

            should_save = False
            save_reason = ""

            if compare_initial_iou == 1:
                if avg_val_iou > best_val_iou:
                    best_val_iou = avg_val_iou
                    should_save = True
                    save_reason = f"Higher IoU found ({best_val_iou:.4f})"
            elif compare_initial_iou == 2:
                should_save = True
                save_reason = "Saving weights for every epoch"
            else:
                if avg_val_weighted_loss < best_val_weighted_loss:
                    best_val_weighted_loss = avg_val_weighted_loss
                    should_save = True
                    save_reason = f"New best weighted loss found ({best_val_weighted_loss:.8f})"

            if should_save:
                save_counter += 1
                base_path, extension = os.path.splitext(best_model_path)
                versioned_model_path = f"{base_path}_{save_counter}{extension}"

                torch.save(model.state_dict(), versioned_model_path)
                print(f"\nModel saved at Epoch {epoch + 1} [{save_reason}] to: {versioned_model_path}\n")

    print(f"\nTraining complete. Final model saved to: {best_model_path} (if not overwritten by versioning)")