from .hub_imports import torch
from .hub_settings import device

from .model import inference

def fgsm_attack(model: any, loss_fn: any, image: torch.Tensor, epsilon: float) -> torch.Tensor:
    
    if epsilon == 0:
        return image
    
    original_image: torch.Tensor = image.clone()
    
    original_image.requires_grad = True
    
    label: int = inference(model, original_image)[0]
    label = torch.Tensor([label]).long().to(device)
    
    output: torch.Tensor = model(original_image)
    model.zero_grad()
    
    loss: torch.Tensor = loss_fn(output, label)
    loss.backward(retain_graph=True)

    perturbed_image: torch.Tensor = original_image + epsilon * original_image.grad.sign()
    
    del original_image, label, output, loss

    return perturbed_image

def ifgsm_attack(model: any, loss_fn: any, image: torch.Tensor, epsilon: float , alpha: float, iter: int) -> torch.Tensor:
    
    if epsilon == 0:
        return image
    
    original_image: torch.Tensor = image.clone()
    perturbed_image: torch.Tensor = image.clone()

    label: int = inference(model, original_image)[0]
    label = torch.Tensor([label]).long().to(device)    
        
    for _ in range(iter):

        perturbed_image.requires_grad = True
        
        output: torch.Tensor = model(perturbed_image)
        model.zero_grad()
        
        loss: torch.Tensor = loss_fn(output, label)
        loss.backward()
        
        perturbed_image: torch.Tensor = perturbed_image + alpha * perturbed_image.grad.sign()
        perturbed_image = torch.clamp(perturbed_image, perturbed_image - epsilon, perturbed_image + epsilon)

        perturbed_image = perturbed_image.detach()

    del original_image, label, output, loss

    return perturbed_image

def pgd_attack(model: any, loss_fn: any, image: torch.Tensor, epsilon: float, alpha: float, iter: int) -> torch.Tensor:
    
    if epsilon == 0:
        return image
    
    original_image: torch.Tensor = image.clone()
    perturbed_image: torch.Tensor = image.clone()
    
    label: int = inference(model, original_image)[0]
    label = torch.Tensor([label]).long().to(device)
    
    for _ in range(iter):
        
        perturbed_image.requires_grad = True
        
        output: torch.Tensor = model(perturbed_image)
        model.zero_grad()
        
        loss: torch.Tensor = loss_fn(output, label)
        loss.backward()

        adv_image: torch.Tensor = perturbed_image + alpha * perturbed_image.grad.sign()
        noise: torch.Tensor = torch.clamp(adv_image - original_image, -epsilon, epsilon)
        perturbed_image = original_image + noise

        perturbed_image = perturbed_image.detach()
    
    del original_image, label, output, loss, adv_image, noise
    
    return perturbed_image