import os
import shutil
from PIL import Image
import tqdm
from logger import get_logger
from transform_factory import get_transforms

logger = get_logger()

def _prepare_output_dir(output_dir, class_names):
    if os.path.exists(output_dir):
        logger.warning(f"La cartella '{output_dir}' esiste già, la elimino")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    logger.debug(f"Cartella '{output_dir}' creata")
    for cls in class_names:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

def _get_images_to_process(input_dir, class_names, valid_ext):
    images = []
    for cls in class_names:
        cls_path = os.path.join(input_dir, cls)
        for img in os.listdir(cls_path):
            if os.path.splitext(img)[1].lower() in valid_ext:
                images.append((cls, img))
            else:
                logger.debug(f"Immagine saltata per estensione non valida: {img}")
    return images

def create_augmented_dataset(config):
    input_dir = config.input_dir
    output_dir = config.output_dir
    num_aug = config.num_augmented_per_image
    copy_original = config.copy_original
    valid_ext = config.valid_extensions

    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    _prepare_output_dir(output_dir, class_names)
    images = _get_images_to_process(input_dir, class_names, valid_ext)

    transform = get_transforms()
    total_aug = 0

    logger.info("Inizio processamento immagini")
    logger.debug(f"Numero di augmentazioni per immagine: {num_aug}")
    logger.info(f"Copia immagini originali: {copy_original}")

    for cls, img_name in tqdm.tqdm(images, desc="Elaborazione immagini"):
        in_path = os.path.join(input_dir, cls, img_name)
        out_dir = os.path.join(output_dir, cls)

        try:
            if copy_original:
                shutil.copy(in_path, os.path.join(out_dir, img_name))

            image = Image.open(in_path).convert("RGB")
            for i in range(num_aug):
                aug_img = transform(image)
                aug_img.save(os.path.join(out_dir, f"aug_{i}_{img_name}"))
                total_aug += 1

        except Exception as e:
            logger.warning(f"Errore con l'immagine '{in_path}': {e}")

    if copy_original:
        logger.debug(f"Copiate {len(images)} immagini originali.")
    logger.debug(f"Generate {total_aug} immagini augmentate.")
    logger.info(f"Augmentazione completata: {len(images) + (total_aug if copy_original else 0)} immagini totali.")
