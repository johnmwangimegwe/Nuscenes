import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, base_detector, conf_thresh):
        super(Network, self).__init__()
        self.base_detector = base_detector
        self.conf_thresh = conf_thresh
        
        # Radar-specific feature extractor
        self.radar_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Fusion layer for combining all modalities
        self.fusion_layer = nn.Linear(3 * 512, 128)  # Example dimensions; adjust based on actual feature sizes

    def forward(self, images, lidar_data, radar_data, model_mode=0):
        if model_mode == 1:  # LiDAR-only mode
            lidar_features = self.lidar_feature_extractor(lidar_data)
            return lidar_features
        
        elif model_mode == 2:  # Radar-only mode
            radar_features = self.radar_feature_extractor(radar_data)
            return radar_features
        
        # Image and LiDAR processing remains unchanged
        img_features = self.base_detector(images)
        lidar_features = self.lidar_feature_extractor(lidar_data)
        
        # Radar feature extraction
        radar_features = self.radar_feature_extractor(radar_data)
        
        # Concatenate features from all modalities
        fused_features = torch.cat((img_features, lidar_features, radar_features), dim=1)
        
        # Apply fusion layer
        output = self.fusion_layer(fused_features)
        return output