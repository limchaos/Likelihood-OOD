import timm
import torch.nn as nn

class TimmModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.name = model_name
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(self.model)   
        self.transform = timm.data.create_transform(**data_config, is_training=True)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model(x)
        return x