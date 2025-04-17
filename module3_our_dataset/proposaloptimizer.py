import torch
import torch.nn as nn

class ProposalOptimizer(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(ProposalOptimizer, self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, 3),  # Generate weights for image, LiDAR, and radar
            nn.Softmax(dim=1)          # Normalize weights
        )

    def forward(self, img_confidence, lidar_confidence, radar_confidence):
        # Generate adaptive weights
        weights = self.weight_generator(torch.cat([img_confidence, lidar_confidence, radar_confidence], dim=1))
        
        # Compute weighted sum of confidences
        combined_confidence = (weights[:, 0].unsqueeze(-1) * img_confidence +
                               weights[:, 1].unsqueeze(-1) * lidar_confidence +
                               weights[:, 2].unsqueeze(-1) * radar_confidence)
        
        return combined_confidence