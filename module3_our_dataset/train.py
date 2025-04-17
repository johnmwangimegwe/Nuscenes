import argparse
import torch
from my_models import Network
from yolov3.models import define_yolo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/2_ckpt_best.pth")
    parser.add_argument("--yolo_cfg", type=str, default="config/yolov3-tiny-12.cfg")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--model_mode", type=int, default=3)  # Three modes: [millieye, yolo, radar]
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_detector = define_yolo(opt.yolo_cfg)
    model = Network(base_detector, opt.conf_thresh).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(opt.checkpoint))
    model.eval()

    # Training loop goes here
    print("Training with three-modal fusion...")

if __name__ == "__main__":
    main()