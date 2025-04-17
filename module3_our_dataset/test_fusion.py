import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from my_models import Network, define_yolo
from train_nuscenes import NuScenesDataset, custom_collate


def process_annotations(annotations):
    """
    Process annotations (labels) into a tensor format suitable for evaluation.
    Args:
        annotations: List of annotation data (e.g., category indices).
    Returns:
        torch.Tensor: Processed labels as a tensor.
    """
    # Example: Assuming annotations are already category indices
    return torch.tensor(annotations, dtype=torch.long)


def evaluate_model():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate multi-modal fusion model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/nuscenes_model_epoch_1.pth",
                        help="Path to the model checkpoint")
    parser.add_argument("--yolo_cfg", type=str, default="config/yolov3-tiny-12.cfg",
                        help="YOLO configuration file path")
    parser.add_argument("--dataset_root", type=str, default="./data",
                        help="Root directory of NuScenes dataset")
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                        help="Confidence threshold for detection")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--save_confusion", action="store_true",
                        help="Save confusion matrix plot")
    opt = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model initialization
    print("Initializing model...")
    base_detector = define_yolo(opt.yolo_cfg).to(device)
    base_detector.eval()
    model = Network(base_detector, opt.conf_thresh).to(device)

    # Load checkpoint
    try:
        model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
        print(f"Successfully loaded weights from {opt.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])

    print(f"Loading {opt.split} dataset...")
    dataset = NuScenesDataset(
        root_dir=opt.dataset_root,
        split=opt.split,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )

    # Evaluation metrics
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Evaluating on {len(dataset)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            images = batch["image"].to(device)
            lidar = batch["lidar_data"].to(device)
            radar = batch["radar_data"].to(device)
            targets = process_annotations(batch["label"]).to(device)

            # Forward pass
            outputs = model(images, lidar, radar)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Store for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * opt.batch_size}/{len(dataset)} samples")

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print("\nEvaluation Results:")
    print(f"- Average Loss: {avg_loss:.4f}")
    print(f"- Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if opt.save_confusion:
        plt.savefig("confusion_matrix.png")
        print("Saved confusion matrix to confusion_matrix.png")
    else:
        plt.show()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    evaluate_model()