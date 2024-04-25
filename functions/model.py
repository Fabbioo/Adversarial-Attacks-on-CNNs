from .hub_imports import torch, torchvision, ResNet50_Weights, resnet50
from .hub_settings import device

def load_model_resnet50() -> torchvision.models:
    
    model: any = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2) # Lista delle classi: https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
    model.eval()
    model.to(device)
    
    return model

def inference(model: any, image: torch.Tensor) -> tuple[int, str, float]:
    
    output: any = model(image).squeeze(0).softmax(0)
    
    class_id: int = output.argmax().item()
    class_name: str = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][class_id]
    class_conf: float = output[class_id].item()

    del output
    
    return (class_id, class_name, class_conf)