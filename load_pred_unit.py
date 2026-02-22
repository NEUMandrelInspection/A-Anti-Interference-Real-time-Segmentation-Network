import os
import json
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loss_unit import calculate_dice, calculate_iou


def convert_json_to_yolo_txt(json_path, image_path):
    """
    Converts a single LabelMe JSON file into a YOLOv8 segmentation format TXT file.
    If successful, creates a .txt file with the same name and returns True.
    """
    txt_path = os.path.splitext(json_path)[0] + '.txt'

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Cannot read image {image_path} during conversion. Skipping.")
        return False
    h, w = image.shape[:2]

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Error reading or parsing JSON file {json_path}: {e}. Skipping.")
        return False

    yolo_lines = []
    class_index = 0
    for shape in data.get('shapes', []):
        points = np.array(shape['points'])
        points[:, 0] /= w
        points[:, 1] /= h
        normalized_coords = points.flatten().tolist()
        line = f"{class_index} " + " ".join(map(str, normalized_coords))
        yolo_lines.append(line)

    if yolo_lines:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
        print(f"Conversion successful: {os.path.basename(json_path)} -> {os.path.basename(txt_path)}")
        return True
    else:
        print(f"Warning: No valid annotations found in {json_path}. TXT file not generated.")
        return False


class CrackDataset(Dataset):
    """
    Dataset class for segmentation and saliency detection.
    Supports data augmentation and dynamic resizing.
    """

    def __init__(self, data_infos, transform=None, use_rgb_input=0, img_size=512, augment=False):
        self.data_infos = data_infos
        self.transform = transform
        self.use_rgb_input = use_rgb_input
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        image_path = info['image_path']
        label_path = info.get('label_path')
        category = info['category']

        sample_subtype = 'none'

        if self.use_rgb_input == 1:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")

        h_orig, w_orig = image.shape[:2] if self.use_rgb_input == 1 else image.shape

        image_resized = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        h_new, w_new = self.img_size, self.img_size

        if self.use_rgb_input == 1:
            image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
        else:
            image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0) / 255.0

        mask = np.zeros((h_new, w_new), dtype=np.uint8)

        # Parse labels based on file extension
        if label_path and os.path.exists(label_path):
            if label_path.endswith('.json'):
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data['shapes']:
                    points = np.array(shape['points'], dtype=np.float32)
                    points[:, 0] *= (w_new / w_orig)
                    points[:, 1] *= (h_new / h_orig)
                    points = np.clip(points, [0, 0], [w_new - 1, h_new - 1]).astype(np.int32)
                    cv2.fillPoly(mask, [points], color=1)

            elif label_path.endswith('.txt'):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        normalized_coords = np.array([float(p) for p in parts[1:]])
                        points_normalized = normalized_coords.reshape(-1, 2)
                        points_pixel = points_normalized * np.array([w_new, h_new])
                        points_pixel = points_pixel.astype(np.int32)
                        cv2.fillPoly(mask, [points_pixel], color=1)

            elif label_path.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg')):
                mask_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_img = cv2.resize(mask_img, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
                    _, mask_binary = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)
                    mask = mask_binary.astype(np.uint8)

        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # Apply random data augmentation (Flip/Rotate)
        if self.augment:
            if torch.rand(1) < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if torch.rand(1) < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                image_tensor = torch.rot90(image_tensor, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask, category, sample_subtype


def draw_true_masks(label_path, img_display):
    """
    Renders the ground truth masks onto the original image for visual comparison.
    """
    h_orig, w_orig, _ = img_display.shape
    true_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    if not label_path or not os.path.exists(label_path):
        return img_display, true_mask

    if label_path.endswith('.json'):
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(true_mask, [points], 255)

    elif label_path.endswith('.txt'):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                normalized_coords = np.array([float(p) for p in parts[1:]])
                points_normalized = normalized_coords.reshape(-1, 2)
                points_pixel = (points_normalized * np.array([w_orig, h_orig])).astype(np.int32)
                cv2.fillPoly(true_mask, [points_pixel], 255)

    elif label_path.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg')):
        mask_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            mask_resized = cv2.resize(mask_img, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            true_mask = mask_binary.astype(np.uint8)

    img_true_label = img_display.copy()
    red_color = np.array([0, 0, 255])
    mask_area = true_mask == 255
    img_true_label[mask_area] = (img_true_label[mask_area] * 0.6 + red_color * 0.4).astype(np.uint8)

    return img_true_label, true_mask


def collect_data_from_dir(directory, category, use_yolo_format=False):
    """
    Collects image and label paths from a given directory for legacy dataset structures.
    """
    data_infos = []
    format_name = "YOLO (.txt)" if use_yolo_format else "LabelMe (.json)"
    print(f"Reading {format_name} formatted data from directory '{directory}'...")

    if not os.path.exists(directory):
        print(f"Error: Specified directory does not exist: {directory}")
        return data_infos

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    for image_path in image_files:
        base_name = os.path.splitext(image_path)[0]
        txt_path = base_name + '.txt'
        json_path = base_name + '.json'
        label_path_to_use = None

        if use_yolo_format:
            if os.path.exists(txt_path):
                label_path_to_use = txt_path
            elif os.path.exists(json_path):
                print(f"File {os.path.basename(txt_path)} not found, but {os.path.basename(json_path)} exists. Attempting conversion...")
                if convert_json_to_yolo_txt(json_path, image_path):
                    label_path_to_use = txt_path
                else:
                    print(f"Warning: Conversion failed for {os.path.basename(json_path)}. Skipping image {os.path.basename(image_path)}.")
                    continue
            else:
                print(f"Warning: Image {os.path.basename(image_path)} is missing .txt or .json annotation. Skipped.")
                continue
        else:
            if os.path.exists(json_path):
                label_path_to_use = json_path
            else:
                print(f"Warning: Label file {os.path.basename(json_path)} for image {os.path.basename(image_path)} does not exist. Skipped.")
                continue

        if label_path_to_use:
            info = {'image_path': image_path, 'label_path': label_path_to_use, 'category': category}
            data_infos.append(info)

    if not data_infos:
        print(f"Warning: No valid annotated data found in directory {directory}.")

    return data_infos


def collect_image_only_data(directory, category):
    """
    Collects image paths for datasets that do not contain annotations (e.g., normal samples).
    """
    data_infos = []
    print(f"Reading image-only data from directory '{directory}'...")
    if not os.path.exists(directory):
        print(f"Warning: Specified directory does not exist: {directory}")
        return data_infos
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    for image_path in image_files:
        info = {'image_path': image_path, 'label_path': None, 'category': category}
        data_infos.append(info)

    if not data_infos:
        print(f"Warning: No image files found in directory {directory}.")

    return data_infos


def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detect_dataset_format(root_dir):
    """
    Automatically detects the dataset structure (Standard Mask, Standard YOLO, or Legacy).
    """
    if not os.path.exists(root_dir):
        return 'unknown'

    images_train_dir = os.path.join(root_dir, 'images', 'train')
    labels_train_dir = os.path.join(root_dir, 'labels', 'train')

    if os.path.isdir(images_train_dir) and os.path.isdir(labels_train_dir):
        try:
            files = os.listdir(labels_train_dir)
            if not files:
                return 'unknown'

            if any(f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg', '.tif')) for f in files):
                return 'standard_mask'

            if any(f.endswith('.txt') for f in files):
                return 'standard_yolo'

        except OSError:
            pass

    return 'legacy'


def load_standard_structure_data(root_dir, split_name, label_ext_candidates):
    """
    Loads data adhering to the standard standard directory structure (images/ and labels/).
    """
    images_dir = os.path.join(root_dir, 'images', split_name)
    labels_dir = os.path.join(root_dir, 'labels', split_name)

    print(f"Loading standard structure dataset ({split_name})")
    print(f"  - Images directory: {images_dir}")
    print(f"  - Labels directory: {labels_dir}")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return []

    if isinstance(label_ext_candidates, str):
        label_ext_candidates = [label_ext_candidates]

    valid_img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in valid_img_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    data_infos = []
    matched_count = 0

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]

        found_label_path = None
        for ext in label_ext_candidates:
            if not ext.startswith('.'):
                ext = '.' + ext
            candidate_path = os.path.join(labels_dir, basename + ext)
            if os.path.exists(candidate_path):
                found_label_path = candidate_path
                break

        info = {
            'image_path': img_path,
            'label_path': found_label_path,
            'category': 'target' if found_label_path else 'normal'
        }

        data_infos.append(info)
        if found_label_path:
            matched_count += 1

    print(f"  - Found {len(data_infos)} images, {matched_count} matched with label files")
    return data_infos


def load_and_prepare_data(config):
    """
    Main entry point for detecting data structures and preparing PyTorch DataLoaders.
    """
    print("\n--- 1. Data Loading and Preparation ---")

    use_rgb = config.get('use_rgb_input', 0)
    batch_size = config['batch_size']
    root_dir = config['source_data_dir']
    use_aug = config.get('use_data_augmentation', 0) == 1

    dataset_format = detect_dataset_format(root_dir)
    print(f"Automatically detected dataset format: {dataset_format}")

    train_data_infos = []
    val_data_infos = []

    if dataset_format == 'standard_mask':
        print(f"--- Enabling Standard Segmentation Dataset Mode (Image Masks) ---")
        mask_exts = ['.png', '.jpg', '.bmp']
        train_data_infos = load_standard_structure_data(root_dir, 'train', mask_exts)
        val_data_infos = load_standard_structure_data(root_dir, 'val', mask_exts)

    elif dataset_format == 'standard_yolo':
        print(f"--- Enabling Standard YOLO Dataset Mode (.txt Labels) ---")
        train_data_infos = load_standard_structure_data(root_dir, 'train', ['.txt'])
        val_data_infos = load_standard_structure_data(root_dir, 'val', ['.txt'])

    else:
        print(f"--- Enabling Legacy Dataset Mode (Custom Folder Structure) ---")
        use_yolo_txt = config.get('use_yolo_txt_format', 0) == 1

        train_target_infos = collect_data_from_dir(config['train_data_path'], 'target',
                                                     use_yolo_format=use_yolo_txt)
        val_target_infos = collect_data_from_dir(config['val_data_path'], 'target', use_yolo_format=use_yolo_txt)

        if not train_target_infos:
            print(f"Error: Failed to load any training data from {config['train_data_path']}.")
            exit()

        if config.get('include_normal_in_training', 1) == 1:
            print("\n--- 'include_normal_in_training' is enabled, loading 'normal' dataset ---")
            train_normal_infos = collect_image_only_data(config['train_normal_path'], 'normal')
            val_normal_infos = collect_image_only_data(config['val_normal_path'], 'normal')
            train_data_infos = train_target_infos + train_normal_infos
            val_data_infos = val_target_infos + val_normal_infos
        else:
            train_data_infos = train_target_infos
            val_data_infos = val_target_infos

    if not train_data_infos:
        print(f"Error: Failed to load any training data from {root_dir}. Please check the directory structure.")
        exit()

    print(f"\nTotal data volume: {len(train_data_infos) + len(val_data_infos)}")
    print(f"Final training set size: {len(train_data_infos)}")
    print(f"Final validation set size: {len(val_data_infos)}")

    if use_aug:
        print(">> Training set data augmentation enabled (Random Flip/Rotate)")
    else:
        print(">> Data augmentation disabled")

    train_dataset = CrackDataset(train_data_infos, use_rgb_input=use_rgb, augment=use_aug)
    val_dataset = CrackDataset(val_data_infos, use_rgb_input=use_rgb, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader, train_data_infos, val_data_infos


def _generate_comparison_image(pred_masks_batch, gate_values_batch, dice_scores_batch, original_images_info,
                               inference_dir, save_false_positives_only=0, draw_bbox_on_inference=0, mask_output_only=0):
    """
    Internal helper function to generate and save comparative visualizations or binary masks during inference.
    """
    import numpy as np
    import cv2
    import os

    batch_size = pred_masks_batch.shape[0]

    for i in range(batch_size):
        info = original_images_info[i]
        category = info['category']
        pred_mask_original_size = pred_masks_batch[i].astype(np.uint8)

        gate_value = gate_values_batch[i][0]
        dice_score = dice_scores_batch[i]
        has_prediction = np.any(pred_mask_original_size)

        if save_false_positives_only == 1:
            is_false_positive = ((category == 'normal' and has_prediction)
                                 or (category == 'target' and not has_prediction))
            if not is_false_positive:
                continue

        image_path = info['image_path']
        label_path = info.get('label_path')

        original_image_bgr = cv2.imread(str(image_path))
        if original_image_bgr is None:
            print(f"Warning: Cannot read image {image_path} in batch process. Skipped.")
            continue
        h, w, _ = original_image_bgr.shape

        pred_mask_binary = cv2.resize(
            pred_mask_original_size, (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        pred_mask_binary = (pred_mask_binary > 0).astype(np.uint8)

        output_filename = os.path.basename(image_path)
        output_path = os.path.join(inference_dir, output_filename)

        if mask_output_only == 1:
            mask_to_save = pred_mask_binary * 255
            cv2.imwrite(output_path, mask_to_save)
            continue

        annotated_pred_image = original_image_bgr.copy()
        green_color = np.array([0, 255, 0])
        mask_area = pred_mask_binary == 1
        annotated_pred_image[mask_area] = (annotated_pred_image[mask_area] * 0.6 + green_color * 0.4).astype(np.uint8)

        if has_prediction and draw_bbox_on_inference == 1:
            contours, _ = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                all_points = np.concatenate(contours, axis=0)
                x, y, w_box, h_box = cv2.boundingRect(all_points)
                cv2.rectangle(annotated_pred_image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                confidence_text = f"Conf: {gate_value:.2f}"
                text_y = y - 10 if y - 10 > 10 else y + h_box + 20
                (text_w, text_h), _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_pred_image, (x, text_y - text_h - 5), (x + text_w, text_y + 5), (0, 255, 0),
                              -1)
                cv2.putText(annotated_pred_image, confidence_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2, cv2.LINE_AA)

        annotated_true_image, _ = draw_true_masks(label_path, original_image_bgr.copy())

        dice_text = f"Dice: {dice_score:.4f}"
        (text_w, text_h), baseline = cv2.getTextSize(dice_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = w - text_w - 10
        text_y = 30
        cv2.putText(annotated_true_image, dice_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        gap = np.full((h, 10, 3), 200, dtype=np.uint8)
        combined_image = np.hstack((original_image_bgr, gap, annotated_pred_image, gap, annotated_true_image))

        cv2.imwrite(output_path, combined_image)


def perform_inference(model, data_infos, output_dir, device, set_name, batch_size, monitor_criterions, compute_only=0,
                      save_false_positives_only=0, draw_bbox_on_inference=1, mask_output_only=0):
    """
    Executes inference on the provided dataset, computes metrics, and selectively generates output visuals.
    """
    print(f"\n--- Running batch inference for {set_name} (Batch Size: {batch_size}) ---")

    if compute_only == 0:
        if mask_output_only == 1:
            print("--- Current Mode: Outputting binary mask images only ---")
        elif save_false_positives_only == 1:
            print("--- Current Mode: Saving comparison images for False Positive samples only ---")

    inference_dir = os.path.join(output_dir, f"{set_name}_comparison_results")
    if compute_only == 0:
        os.makedirs(inference_dir, exist_ok=True)

    if not data_infos:
        print(f"No data found for inference on {set_name}.")
        return

    inference_dataset = CrackDataset(data_infos)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model.eval()

    running_metrics = {k: 0.0 for k in ['bce', 'dice', 'iou', 'mse']}
    all_dice_scores = []
    all_iou_scores = []

    print("Step 1/2: Executing model inference (computation only)...")
    all_predictions = []
    all_gate_values = []
    all_dice_values_for_viz = []

    with torch.no_grad():
        for images_batch, masks_batch, categories_batch, subtypes_batch in tqdm(inference_dataloader,
                                                                                desc=f"Computing {set_name}"):
            images_batch, masks_batch = images_batch.to(device, non_blocking=True), masks_batch.to(device)
            pred_tensors_batch, gate_values_batch = model(images_batch)

            has_labeled_data = torch.any(masks_batch.sum(dim=[1, 2, 3]) > 0)
            if has_labeled_data:
                running_metrics['bce'] += monitor_criterions['bce'](pred_tensors_batch, masks_batch).item()
                running_metrics['dice'] += monitor_criterions['dice'](pred_tensors_batch, masks_batch).item()
                running_metrics['iou'] += monitor_criterions['iou'](pred_tensors_batch, masks_batch).item()
                running_metrics['mse'] += monitor_criterions['mse'](pred_tensors_batch, masks_batch).item()

            pred_mask_binary_batch = (pred_tensors_batch > 0.5)
            pred_mask_numpy = pred_mask_binary_batch.squeeze(1).cpu().numpy()
            true_mask_numpy = masks_batch.squeeze(1).cpu().numpy()

            batch_dice_viz_temp = []

            for i in range(pred_mask_numpy.shape[0]):
                if np.any(true_mask_numpy[i]):
                    d_val = calculate_dice(pred_mask_numpy[i], true_mask_numpy[i])
                    all_dice_scores.append(d_val)
                    all_iou_scores.append(calculate_iou(pred_mask_numpy[i], true_mask_numpy[i]))

                if np.any(true_mask_numpy[i]):
                    d_viz = calculate_dice(pred_mask_numpy[i], true_mask_numpy[i])
                else:
                    if not np.any(pred_mask_numpy[i]):
                        d_viz = 1.0
                    else:
                        d_viz = 0.0
                batch_dice_viz_temp.append(d_viz)

            if compute_only == 0:
                all_predictions.append(pred_mask_binary_batch.float().squeeze(1).cpu().numpy())
                all_gate_values.append(gate_values_batch.cpu().numpy())
                all_dice_values_for_viz.append(batch_dice_viz_temp)

    num_batches = len(inference_dataloader)
    avg_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
    avg_iou = np.mean(all_iou_scores) if all_iou_scores else 0.0

    prefix = "Validation" if "validation" in set_name else "Training" if "training" in set_name else set_name

    print(f"\n--- {prefix} Set Inference Results ---")
    print(f"{prefix} Loss - BCE: {running_metrics['bce'] / num_batches:.6f}, "
          f"Dice: {running_metrics['dice'] / num_batches:.6f}, "
          f"IoU: {running_metrics['iou'] / num_batches:.6f}, "
          f"MSE: {running_metrics['mse'] / num_batches:.6f}")
    print(f"{prefix} Average Dice (Labeled samples only): {avg_dice:.4f}, {prefix} Average IoU (Labeled samples only): {avg_iou:.4f}\n")

    if compute_only == 1:
        print(f"--- Pure computation for {set_name} complete. Skipping image saving. ---")
        return

    print("Step 2/2: Generating and saving results (I/O intensive)...")
    processed_count = 0
    for i, pred_masks_batch in enumerate(tqdm(all_predictions, desc=f"Saving {set_name} Results")):
        gate_values_batch = all_gate_values[i]
        dice_scores_batch = all_dice_values_for_viz[i]
        current_batch_size = pred_masks_batch.shape[0]
        original_images_info_batch = data_infos[processed_count: processed_count + current_batch_size]

        _generate_comparison_image(
            pred_masks_batch,
            gate_values_batch,
            dice_scores_batch,
            original_images_info_batch,
            inference_dir,
            save_false_positives_only,
            draw_bbox_on_inference=draw_bbox_on_inference,
            mask_output_only=mask_output_only
        )

        processed_count += current_batch_size

    print(f"--- Inference results generation for {set_name} complete ---")