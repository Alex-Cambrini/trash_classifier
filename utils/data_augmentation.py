import os
import shutil
from PIL import Image
from torchvision import transforms
import tqdm
from logger import get_logger

logger = get_logger()

AUGMENTATIONS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

def create_augmented_dataset(input_dir, output_dir, num_augmented_per_image=3):
    logger.info(f"Creo dataset aumentato da '{input_dir}' a '{output_dir}' con {num_augmented_per_image} immagini per originale")

    if os.path.exists(output_dir):
        logger.debug(f"Cartella '{output_dir}' esiste già, la elimino")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.debug(f"Cartella '{output_dir}' creata")

    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    all_images = []
    for class_name in class_names:
        class_input_path = os.path.join(input_dir, class_name)
        for img_name in os.listdir(class_input_path):
            all_images.append((class_name, img_name))

    total_augmented = 0
    for class_name, img_name in tqdm.tqdm(all_images, desc="Processing images"):
        try:
            class_input_path = os.path.join(input_dir, class_name)
            class_output_path = os.path.join(output_dir, class_name)

            img_path = os.path.join(class_input_path, img_name)
            # Copia immagine originale
            shutil.copy(img_path, os.path.join(class_output_path, img_name))

            image = Image.open(img_path).convert("RGB")
            
            for i in range(num_augmented_per_image):
                augmented_img = AUGMENTATIONS(image)
                output_img_path = os.path.join(class_output_path, f"aug_{i}_{img_name}")
                augmented_img.save(output_img_path)
                total_augmented += 1

        except Exception as e:
            logger.warning(f"Errore con immagine '{img_path}': {e}")

    logger.info(f"Augmentation completata con successo: {total_augmented} immagini salvate.")
