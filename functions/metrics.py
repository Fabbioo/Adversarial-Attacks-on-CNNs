from .hub_imports import np, read_image

from .attacks import *
from .model import inference
from .utility import preprocess, postprocess

def compute_accuracy(attack_type: int, dataset: list[str], model: any, loss_fn: any, epsilons: list[float], alphas: list[float], iters: int = 0) -> tuple[list, dict]:
    
    accuracies: list[int] = []
    dict_wrong_preds: dict = {}
    
    for epsilon, alpha in zip(epsilons, alphas):
        
        correct_predicts: int = 0
        wrong_preds: list = []
        
        for image in dataset:
            
            original_image: torch.Tensor = read_image(image)
            original_image = preprocess(original_image)
            
            match attack_type:
                case "FGSM":
                    perturbed_image: torch.Tensor = fgsm_attack(model, loss_fn, original_image, epsilon)
                case "I-FGSM":
                    perturbed_image: torch.Tensor = ifgsm_attack(model, loss_fn, original_image, epsilon, alpha, iters)
                case "PGD":
                    perturbed_image: torch.Tensor = pgd_attack(model, loss_fn, original_image, epsilon, alpha, iters)
                case _:
                    raise ValueError("Invalid attack type")
            
            original_image = postprocess(original_image)
            perturbed_image = postprocess(perturbed_image)
            
            pred1: int = inference(model, original_image)[0]
            pred2: int = inference(model, perturbed_image)[0]
            if pred1 == pred2:
                correct_predicts += 1
            else:
                wrong_preds.append(perturbed_image)
            
            del image, original_image, perturbed_image, pred1, pred2
        
        correct_predicts /= len(dataset)
        accuracies.append(correct_predicts)
        for _ in range(5):
            np.random.shuffle(wrong_preds)
        dict_wrong_preds[epsilon] = wrong_preds

        del correct_predicts, wrong_preds, epsilon, alpha
        
    return (accuracies, dict_wrong_preds)