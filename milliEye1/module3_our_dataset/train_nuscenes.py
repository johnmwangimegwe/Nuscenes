import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Import custom modules
from my_models import Network, define_yolo
from proposaloptimizer import ProposalOptimizer

# NuScenes category mapping (23 classes)
NUSCENES_CATEGORIES = [
    'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'movable_object.barrier',
    'movable_object.debris', 'movable_object.pushable_pullable', 'movable_object.trafficcone',
    'static_object.bicycle_rack', 'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid',
    'vehicle.car', 'vehicle.construction', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck',
    'flat.driveable_surface', 'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade',
    'static.vegetation'
]

# Custom Dataset Class
class NuScenesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the NuScenes dataset directory.
            split (str): "train", "val", or "test".
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.category_to_idx = {cat: idx for idx, cat in enumerate(NUSCENES_CATEGORIES)}
        self.samples = self._load_metadata()

    def _load_metadata(self):
        """Load metadata for the dataset with proper scene token mapping."""
        try:
            samples = []
            sample_file = os.path.join(self.root_dir, "v1.0-mini", "sample.json")
            scene_file = os.path.join(self.root_dir, "v1.0-mini", "scene.json")
            annotation_file = os.path.join(self.root_dir, "v1.0-mini", "sample_annotation.json")

            # Load scene data
            with open(scene_file, "r") as f:
                scenes = json.load(f)

            # Load sample data
            with open(sample_file, "r") as f:
                all_samples = json.load(f)

            # Load annotations
            with open(annotation_file, "r") as f:
                annotations = json.load(f)

            scene_splits = {
                "train": ["scene-0061", "scene-0553", "scene-0655"],
                "val": ["scene-0757", "scene-1077"],
                "test": ["scene-0916", "scene-1100"]
            }

            valid_scenes = scene_splits[self.split]
            for sample in all_samples:
                scene_token = sample.get("scene_token")
                scene_name = next((s["name"] for s in scenes if s["token"] == scene_token), None)
                if scene_name in valid_scenes:
                    # Validate the presence of required keys
                    if "filename" not in sample:
                        print(f"Warning: Missing 'filename' key in sample {sample}. Skipping...")
                        continue

                    image_path = os.path.join(self.root_dir, "samples", "CAM_FRONT", sample["filename"])
                    lidar_path = os.path.join(self.root_dir, "samples", "LIDAR_TOP", sample["lidar_filename"])
                    radar_path = os.path.join(self.root_dir, "samples", "RADAR_FRONT", sample["radar_filename"])

                    # Ensure all files exist
                    if not os.path.exists(image_path):
                        print(f"Warning: Image file not found: {image_path}. Skipping...")
                        continue

                    if not os.path.exists(lidar_path):
                        print(f"Warning: LiDAR file not found: {lidar_path}. Skipping...")
                        continue

                    if not os.path.exists(radar_path):
                        print(f"Warning: Radar file not found: {radar_path}. Skipping...")
                        continue

                    # Get annotations for the sample
                    sample_token = sample["token"]
                    sample_annotations = [anno for anno in annotations if anno["sample_token"] == sample_token]
                    if not sample_annotations:
                        print(f"Warning: No annotations found for sample {sample_token}. Skipping...")
                        continue

                    # Add sample to dataset
                    samples.append({
                        "image_path": image_path,
                        "lidar_path": lidar_path,
                        "radar_path": radar_path,
                        "annotations": sample_annotations
                    })

            return samples
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            image_path = sample["image_path"]
            lidar_path = sample["lidar_path"]
            radar_path = sample["radar_path"]

            # Load and preprocess data
            image = Image.open(image_path).convert("RGB")
            lidar_data = np.load(lidar_path)  # Example: Load LiDAR point cloud
            radar_data = np.load(radar_path)  # Example: Load radar data

            if self.transform:
                image = self.transform(image)

            # Extract label from annotations
            labels = [self.category_to_idx[anno["category_name"]] for anno in sample["annotations"]]
            label = torch.tensor(labels[0], dtype=torch.long)  # Use the first annotation as the label

            return {
                "image": image,
                "lidar_data": torch.tensor(lidar_data, dtype=torch.float32),
                "radar_data": torch.tensor(radar_data, dtype=torch.float32),
                "label": label
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return None

# Custom Collate Function
def custom_collate(batch):
    """Custom collate function to handle variable-sized LiDAR and radar data."""
    batch = [item for item in batch if item is not None]  # Skip invalid samples
    if len(batch) == 0:
        return None

    images = torch.stack([item['image'] for item in batch])
    lidar_data = torch.stack([item['lidar_data'] for item in batch])
    radar_data = torch.stack([item['radar_data'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])

    return {"image": images, "lidar_data": lidar_data, "radar_data": radar_data, "label": labels}

# Training Function
def train_nuscenes(opt):
    """Train the multi-modal fusion model on the NuScenes dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize base detector (YOLO)
    base_detector = define_yolo(opt.yolo_cfg).to(device)

    # Initialize the main model
    model = Network(base_detector, opt.conf_thresh, num_classes=len(NUSCENES_CATEGORIES)).to(device)

    # Weight freezing (retain only early layers frozen)
    for name, param in model.named_parameters():
        if "base_detector" in name and any(f"conv_{i}" in name for i in range(5)):
            param.requires_grad = False
            print(f"Frozen: {name}")

    # Proposal Optimization Module
    proposal_optimizer = ProposalOptimizer(num_classes=len(NUSCENES_CATEGORIES), feature_dim=512).to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(proposal_optimizer.parameters()),
        lr=opt.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NuScenesDataset(root_dir=opt.dataset_root, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)

    # Training Loop
    for epoch in range(opt.epochs):
        model.train()
        proposal_optimizer.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch is None:  # Skip empty batches
                continue

            # Move data to device
            images = batch["image"].to(device)
            lidar_data = batch["lidar_data"].to(device)
            radar_data = batch["radar_data"].to(device)
            labels = batch["label"].to(device)

            # Forward pass through the model
            fused_features = model(images, lidar_data, radar_data)

            # Proposal optimization
            img_confidence = fused_features[:, :128]  # Example: Split features for each modality
            lidar_confidence = fused_features[:, 128:256]
            radar_confidence = fused_features[:, 256:]
            optimized_features = proposal_optimizer(img_confidence, lidar_confidence, radar_confidence)

            # Compute loss
            outputs = optimized_features
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            running_loss += loss.item()
            if batch_idx % opt.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{opt.epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch+1}.pth")
        print(f"Saved checkpoint at epoch {epoch+1}")

        # Learning rate scheduling
        scheduler.step(avg_loss)

    print("Training completed.")

# Argument Parsing and Execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NuScenes Multi-Modal Fusion Model")
    parser.add_argument("--dataset_root", default="./data", help="Path to dataset root")
    parser.add_argument("--yolo_cfg", default="config/yolov3.cfg", help="Path to YOLO config file")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")

    opt = parser.parse_args()

    # Create checkpoints directory
    os.makedirs("./checkpoints", exist_ok=True)

    # Train the model
    train_nuscenes(opt)