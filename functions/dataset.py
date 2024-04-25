from .hub_imports import os, shutil
from .hub_settings import working_directory_path

def load_dataset(images_path: str, added_new_images: bool = False) -> list[str]:
    
    if added_new_images: # added_new_images = True solo quando vengono aggiunte nuove immagini al path.
        
        extension: str = ".jpg"
        
        # Per comodità rinomino tutte le immagini su cui fare inferenza con nomi del tipo 1.jpg, 2.jpg, 3.jpg, ...
        files: list[str] = [os.path.join(images_path, elem) for elem in os.listdir(images_path) if not elem.startswith('.')]
        num: int = 0
        for file in files:
            num += 1
            os.rename(file, str(num) + extension)
            del file
        
        # Poichè a seguito della ridenominazione le immagini vengono spostate al di fuori dalla cartella images, le riporto dentro.
        files: list[str] = [os.path.join(working_directory_path, elem) for elem in os.listdir(working_directory_path) if not elem.startswith('.') and elem.endswith(extension)]
        for file in files:
            shutil.move(os.path.join(working_directory_path, file), images_path)
            del file
        
        del extension, files, num
    
    # Creo il dataset con tutte le immagini su cui eseguire gli attacchi.
    images: list[str] = [os.path.join(images_path, elem) for elem in os.listdir(images_path) if not elem.startswith('.')]
    dataset: list[str] = [image for image in images]
    
    # Opzionale: ordino le immagini del dataset in base al nome.
    dataset = sorted(dataset, key = lambda x: int(x[x.rfind('/') + 1 : x.rfind('.')]))

    del images
    
    return dataset

def len_dataset(images_path: str) -> int:
    return len(load_dataset(images_path))