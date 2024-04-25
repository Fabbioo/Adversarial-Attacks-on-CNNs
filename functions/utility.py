from .hub_imports import torch, torchvision, np, plt
from .hub_settings import device, img_resize

def tensor2array(tensor: torch.Tensor) -> np.ndarray:
    
    ndarray: np.ndarray = tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
    
    return ndarray.astype("uint8")

def visualize(image: torch.Tensor | np.ndarray) -> None:
    
    if isinstance(image, torch.Tensor):
        image = tensor2array(image)
    
    plt.figure(figsize = (8, 8))
    plt.axis("off")
    plt.imshow(image)

def normalize(image: torch.Tensor) -> torch.Tensor:
    
    normalization = torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    image = normalization(image)

    del normalization
    
    return image

def denormalize(image: torch.Tensor) -> torch.Tensor:
    
    denormalization = torchvision.transforms.Normalize(
        mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std = [1/0.229, 1/0.224, 1/0.255]
    )
    image = denormalization(image)

    del denormalization
    
    return image

def preprocess(image: torch.Tensor) -> torch.Tensor:
    
    image = image.to(device)
    image = torch.clamp(image, 0, 255).to(torch.uint8)
    image = torchvision.transforms.functional.resize(image, [img_resize[0], img_resize[1]])
    image = image.float() / 255.
    image = normalize(image)
    image = image.unsqueeze(0)
    
    return image

def postprocess(image: torch.Tensor) -> torch.Tensor:
    
    image = denormalize(image)
    image = torch.clamp(image, 0, 1)
    
    return image