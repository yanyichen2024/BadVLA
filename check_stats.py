
import timm
import torch

try:
    dino = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=False)
    siglip = timm.create_model("vit_so400m_patch14_siglip_224", pretrained=False)

    dino_cfg = timm.data.resolve_model_data_config(dino)
    siglip_cfg = timm.data.resolve_model_data_config(siglip)

    print("DINOv2 Mean/Std:", dino_cfg['mean'], dino_cfg['std'])
    print("SigLIP Mean/Std:", siglip_cfg['mean'], siglip_cfg['std'])
except Exception as e:
    print(e)
