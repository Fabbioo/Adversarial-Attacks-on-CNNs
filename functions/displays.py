from .hub_imports import torchvision, plt, np, torch, grad_cam, gridspec
from .hub_settings import img_resize

from .model import inference
from .utility import tensor2array

def preds_display(model: any, tripla: tuple, epsilon: float, show_noise: bool = False) -> None:

    outputs_orig: tuple[int, str, float] = inference(model, tripla[0])
    outputs_pert: tuple[int, str, float] = inference(model, tripla[2])
    
    color: str = ""
    if outputs_orig[1] == outputs_pert[1]: # Se le due predizioni coincidono ...
        color = "green" # ... stampo una scritta verde ...
    else:
        color = "red" # ... altrimenti rossa.
    
    if show_noise:
        images: list = [tensor2array(tripla[0]), tensor2array(tripla[1]), tensor2array(tripla[2])]
        objects: list[str] = ["ORIGINAL", "NOISE", "PERTURBED"]
        plt.figure(figsize = (15, 5))
        gs = gridspec.GridSpec(1, 5, width_ratios = [5, 0.1, 5, 0.1, 5])
        for i in range(len(images) + 2): # + 2 perchè voglio rappresentare sia il + sia il =
            plt.subplot(gs[i])
            match i:
                case 0:
                    plt.imshow(images[0])
                    plt.title(objects[0] + "\n\n" + str(outputs_orig[0]) + ": " + outputs_orig[1] + f", {outputs_orig[2] * 100:.3}%", color = "green")
                case 1:
                    plt.text(0.5, 0.5, "+", fontsize = 40, ha = "center")
                case 2:
                    plt.imshow(images[1])
                    plt.title(objects[1])
                case 3:
                    plt.text(0.5, 0.5, "=", fontsize = 40, ha = "center")
                case 4:
                    plt.imshow(images[2])
                    plt.title(objects[2] + f" ($\epsilon$ = {epsilon})\n\n" + str(outputs_pert[0]) + ": " + outputs_pert[1] + f", {outputs_pert[2] * 100:.3}%", color = color)
            plt.axis("off")
    else:
        images: list = [tensor2array(tripla[0]), tensor2array(tripla[2])]
        objects: list[str] = ["ORIGINAL", "PERTURBED"]
        plt.figure()
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i])
            plt.axis("off")
            match i:
                case 0:
                    plt.title(objects[0] + "\n\n" + str(outputs_orig[0]) + ": " + outputs_orig[1] + f", {outputs_orig[2] * 100:.3}%", color = "green")
                case 1:
                    plt.title(objects[1] + f" ($\epsilon$ = {epsilon})\n\n" + str(outputs_pert[0]) + ": " + outputs_pert[1] + f", {outputs_pert[2] * 100:.3}%", color = color)

    del images, objects, outputs_orig, outputs_pert, color, i

def gradcam_display(model: any, tripla: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    
    layer: str = "layer4" # Customize this value: layer1, layer2, layer3, layer4.
    
    titles: list[str] = ["ORIGINAL", "PERTURBED"]
    
    outputs_orig: tuple[int, str, float] = inference(model, tripla[0])
    outputs_pert: tuple[int, str, float] = inference(model, tripla[2])
    color: str = ""
    if outputs_orig[1] == outputs_pert[1]: # Se le due predizioni coincidono ...
        color = "green" # ... stampo una scritta verde ...
    else:
        color = "red" # ... altrimenti rossa.
    
    cam_orig = grad_cam(model, tripla[0], target = outputs_orig[0], saliency_layer = layer)
    cam_orig = (cam_orig - cam_orig.min()) / (cam_orig.max() - cam_orig.min())
    cam_orig = torchvision.transforms.functional.resize(cam_orig, [img_resize[0], img_resize[1]])
    image_to_show_orig = cam_orig[0].permute(1, 2, 0).detach().cpu().numpy()
    
    cam_pert = grad_cam(model, tripla[2], target = outputs_pert[0], saliency_layer = layer)
    cam_pert = (cam_pert - cam_pert.min()) / (cam_pert.max() - cam_pert.min())
    cam_pert = torchvision.transforms.functional.resize(cam_pert, [img_resize[0], img_resize[1]])
    image_to_show_pert = cam_pert[0].permute(1, 2, 0).detach().cpu().numpy()
    
    plt.figure()
    for i in range(len(titles)):
        plt.subplot(1, len(titles), i + 1)
        plt.axis("off")
        match i:
            case 0:
                plt.imshow(image_to_show_orig)
                plt.imshow(tensor2array(tripla[0]), alpha = 0.4)
                plt.title(titles[i] + "\n\n" + str(outputs_orig[0]) + ": " + outputs_orig[1] + f", {outputs_orig[2] * 100:.3}%", color = "green")
            case 1:
                plt.imshow(image_to_show_pert)
                plt.imshow(tensor2array(tripla[2]), alpha = 0.4)
                plt.title(titles[i] + "\n\n" + str(outputs_pert[0]) + ": " + outputs_pert[1] + f", {outputs_pert[2] * 100:.3}%", color = color)

    del layer, titles, outputs_orig, outputs_pert, color, cam_orig, cam_pert, image_to_show_orig, image_to_show_pert, i

def accuracy_display(dataset: list[str], model: any, epsilons: list[float], accuracies: tuple[list[float]], iter: int, wrong_preds: tuple[dict], dict_show_wrong_preds: dict) -> None:
    
    plt.figure()
    plt.plot(epsilons, accuracies[0], label = "FGSM", marker = 'o', color = 'r')
    plt.plot(epsilons, accuracies[1], label = "I-FGSM", marker = 'o', color = 'g')
    plt.plot(epsilons, accuracies[2], label = "PGD", marker = 'o', color = 'b')
    plt.legend(loc = "lower left")
    plt.suptitle(f"Performance del modello ResNet-50 al variare di $\epsilon$ sfruttando iters = {iter}")
    plt.xlabel("$\epsilon$")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, 1.1, step = 0.1))
    plt.yticks(np.arange(0, 1.1, step = 0.1))
    plt.grid()
    
    temp_str: str = ""
    if dict_show_wrong_preds["show_FGSM_wrong_preds"] == True:
        temp_str = "FGSM: "
        wrong_preds_display(wrong_preds[0], dataset, model, epsilons, temp_str, iter)
    if dict_show_wrong_preds["show_IFGSM_wrong_preds"] == True:
        temp_str = "I-FGSM: "
        wrong_preds_display(wrong_preds[1], dataset, model, epsilons, temp_str, iter)
    if dict_show_wrong_preds["show_PGD_wrong_preds"] == True:
        temp_str = "PGD: "
        wrong_preds_display(wrong_preds[2], dataset, model, epsilons, temp_str, iter)
    
    del temp_str

def wrong_preds_display(dict_wrong_preds: dict, dataset: list[str], model: any, epsilons: list[float], temp_str: str, iter: int) -> None:

    column_number: int = 4 # Numero arbitrario di grafici da creare per ciascun valore di epsilon.
    
    # Se non ci sono abbastanza grafici da creare per il particolare valore di epsilon modifico il numero di grafici da creare.
    min_number_of_elements_in_list_for_each_epsilon: int = len(dataset) # Scelta arbitraria di inizializzazione.
    for i in range(len(epsilons)):
        if epsilons[i] == 0: # In corrispondenza di epsilon = 0 non ci possono essere errori nelle predizioni, perciò passo direttamente all'iterazione successiva.
            continue
        len_dict_wrong_preds_epsilons_i: int = len(dict_wrong_preds[epsilons[i]])
        if len_dict_wrong_preds_epsilons_i < min_number_of_elements_in_list_for_each_epsilon:
            min_number_of_elements_in_list_for_each_epsilon = len_dict_wrong_preds_epsilons_i
    if column_number > min_number_of_elements_in_list_for_each_epsilon:
        column_number = min_number_of_elements_in_list_for_each_epsilon
    
    # Plot delle immagini perturbate classificate erroneamente.
    for i in range(len(epsilons)):
        if epsilons[i] == 0: # In corrispondenza di epsilon = 0 non ci possono essere errori nelle predizioni, perciò passo direttamente all'iterazione successiva.
            continue
        plt.figure()
        for j in range(column_number):
            _, class_name, class_conf = inference(model, dict_wrong_preds[epsilons[i]][j])
            plt.suptitle(f"{temp_str} $\epsilon$ = {epsilons[i]}, iters = {iter}")
            plt.subplot(1, column_number, j + 1)
            plt.imshow(tensor2array(dict_wrong_preds[epsilons[i]][j]))
            plt.title("Wrong pred:\n" + class_name + "\n" + f"{class_conf*100:.3}%" + "\n", color = "red")
            plt.axis("off")
    
    del column_number, min_number_of_elements_in_list_for_each_epsilon, len_dict_wrong_preds_epsilons_i, i, j, class_name, class_conf