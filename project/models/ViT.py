import torch
import torchvision
import pathlib, requests, pathlib
import timm 



def download(url, filename):
    data = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(data.content)

def load_model(obj, builder, model_dir_path, weights):
    model_dir_path = pathlib.Path(model_dir_path)
    model_dir_path.mkdir(exist_ok=True, parents=True)
    model_path = model_dir_path / f'{obj.name}.pth'
    if weights not in obj.WEIGTHS:
        raise ValueError(f"weights should be one of {obj.WEIGTHS.keys()}")
    model = builder(weights=obj.WEIGTHS[weights])
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        torch.save(model.state_dict(), model_path)
    return model

TRANSFORM_IMAGENET1K_V1 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

TRANSFORM_IMAGENET1K_SWAG_E2E_V1_384 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((384, 384)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
TRANSFORM_IMAGENET1K_SWAG_E2E_V1_512 = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 512)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

TRANSFORMS = {
    "ViT_B_16_IMAGENET1K_V1": TRANSFORM_IMAGENET1K_V1,
    "ViT_B_16_IMAGENET1K_SWAG_E2E_V1": TRANSFORM_IMAGENET1K_SWAG_E2E_V1_384,
    "ViT_B_16_IMAGENET1K_SWAG_LINEAR_V1": TRANSFORM_IMAGENET1K_V1,
    "ViT_B_32_IMAGENET1K_V1": TRANSFORM_IMAGENET1K_V1,
    "ViT_L_16_IMAGENET1K_V1": TRANSFORM_IMAGENET1K_V1,
    "ViT_L_16_IMAGENET1K_SWAG_E2E_V1": TRANSFORM_IMAGENET1K_SWAG_E2E_V1_512,
    "ViT_L_16_IMAGENET1K_SWAG_LINEAR_V1": TRANSFORM_IMAGENET1K_V1,
    "ViT_L_32_IMAGENET1K_V1": TRANSFORM_IMAGENET1K_V1,
}

class BASE_ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = TRANSFORMS[self.name]

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

class ViT_B_16(BASE_ViT):
    WEIGTHS = {
        "IMAGENET1K_V1": torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1,
        "IMAGENET1K_SWAG_E2E_V1": torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        "IMAGENET1K_SWAG_LINEAR_V1": torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
    }
    def __init__(self, model_dir_path='checkpoints', weights='IMAGENET1K_V1'):
        self.name = f'ViT_B_16_{weights}'
        super().__init__()
        self.model = load_model(self, torchvision.models.vit_b_16, model_dir_path, weights)

 
class ViT_B_32(BASE_ViT):
    WEIGTHS = {
        "IMAGENET1K_V1": torchvision.models.ViT_B_32_Weights.IMAGENET1K_V1,
    }
    def __init__(self, model_dir_path='checkpoints', weights='IMAGENET1K_V1'):
        self.name = f'ViT_B_32_{weights}'
        super().__init__()
        self.model = load_model(self, torchvision.models.vit_b_32, model_dir_path, weights)

  

class ViT_L_16(BASE_ViT):
    WEIGTHS = {
        "IMAGENET1K_V1": torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1,
        "IMAGENET1K_SWAG_E2E_V1": torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1,
        "IMAGENET1K_SWAG_LINEAR_V1": torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1,
    }
    def __init__(self, model_dir_path='checkpoints', weights='IMAGENET1K_V1'):
        self.name = f'ViT_L_16_{weights}'
        super().__init__()
        self.model = load_model(self, torchvision.models.vit_l_16, model_dir_path, weights)

   

class VIT_L_32(BASE_ViT):
    WEIGTHS = {
        "IMAGENET1K_V1": torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1,
    }
    def __init__(self, model_dir_path='checkpoints', weights='IMAGENET1K_V1'):
        super().__init__()
        self.name = f'ViT_L_32_{weights}'
        self.model = load_model(self, torchvision.models.vit_l_32, model_dir_path, weights)
class DinoV2_ViT_B_14(torch.nn.Module):
    def __init__(self, model_dir_path='checkpoints'):
        super().__init__()
        self.name = 'DinoV2_ViT_B_14'
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        weights = pathlib.Path(model_dir_path)/'DinoV2_ViT_B_14.pth'
        if not weights.exists():
            torch.save(self.model.state_dict(), weights)
        else:
            self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model(x)
        return x
 
class ViT_P16_21k(torch.nn.Module):
    def __init__(self, model_dir_path='checkpoints'):
        from mmpretrain.apis import init_model
        import mmengine
        super().__init__()
        self.name = 'ViT_P16_21k'
        model_dir = pathlib.Path(model_dir_path)
        weights = model_dir/'vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
        if not model_dir.exists():
            model_dir.mkdir(exist_ok=True)
        if not weights.exists():
            url = 'https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
            download(url, weights)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        cfg = mmengine.Config(cfg_dict=dict(model=dict(
            type='ImageClassifier',
            backbone=dict(
                type='VisionTransformer',
                arch='b',
                img_size=384,
                patch_size=16,
                drop_rate=0.1,
                init_cfg=[
                    dict(
                        type='Kaiming',
                        layer='Conv2d',
                        mode='fan_in',
                        nonlinearity='linear')
                ]),
            neck=None,
            head=dict(
                type='VisionTransformerClsHead',
                num_classes=1000,
                in_channels=768,
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1,
                    mode='classy_vision'),
            )))
        )

        self.model = init_model(cfg, str(weights), 0)

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model.backbone(x)[0]
        return x
    
class TIMM_ViT_BASE(torch.nn.Module):
    def __init__(self, model_dir_path, weights):
        super().__init__()
        model_dir = pathlib.Path(model_dir_path)
        self.model = timm.create_model(weights, pretrained=True)
        weights = model_dir/f'{weights}.pth'
        data_config = timm.data.resolve_model_data_config(self.model)   
        self.transform = timm.data.create_transform(**data_config, is_training=True)

        if not model_dir.exists():
            model_dir.mkdir(exist_ok=True)
        if not weights.exists():
            torch.save(self.model.state_dict(), weights)
        else:
            self.model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    def forward(self, x):
        x = self.model(x)
        return x
    
    def features(self, x):
        x = self.model(x)
        return x

#vit_base_patch8_224.dino
class ViT_P8_224_Dino(TIMM_ViT_BASE):
    def __init__(self, model_dir_path='checkpoints'):
        self.name = 'ViT_P8_224_Dino'
        super().__init__(model_dir_path,'vit_base_patch8_224.dino')
        

#vit_base_patch14_dinov2
class ViT_P14_DinoV2(TIMM_ViT_BASE):
    def __init__(self, model_dir_path='checkpoints'):
        self.name = 'ViT_P14_DinoV2'
        super().__init__(model_dir_path, "vit_base_patch14_dinov2.lvd142m")
      

#vit_base_patch14_reg4_dinov2.lvd142m
class ViT_P14_DinoV2_Reg4(TIMM_ViT_BASE):
    def __init__(self, model_dir_path='checkpoints'):
        self.name = 'ViT_P14_DinoV2_Reg4'
        super().__init__(model_dir_path, 'vit_base_patch14_reg4_dinov2.lvd142m')

