import os
import shutil
import sys
from collections import Counter
from PIL import Image
import tqdm
from logger import get_logger
from transform_factory import get_transforms

logger = get_logger()


def _prepare_output_dir(output_dir, class_names, overwrite):
    if os.path.exists(output_dir):
        if overwrite:
            logger.info(f"Overwrite abilitato: elimino la cartella esistente '{output_dir}'")
            shutil.rmtree(output_dir)
        else:
            logger.error(
                f"La cartella '{output_dir}' esiste già. "
                f"Imposta overwrite=true nel config per sovrascrivere."
            )
            sys.exit(1)

    os.makedirs(output_dir)
    logger.debug(f"Cartella '{output_dir}' creata")
    for cls in class_names:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)


def _get_images_to_process(input_dir, class_names, valid_ext):
    images = []
    for cls in class_names:
        cls_path = os.path.join(input_dir, cls)
        for img_name in os.listdir(cls_path):
            if os.path.splitext(img_name)[1].lower() in valid_ext:
                img_path = os.path.join(cls_path, img_name)
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    images.append((cls, img_name))
                except Exception:
                    logger.warning(f"Immagine corrotta o non apribile: {img_path}")
            else:
                logger.debug(f"Immagine saltata per estensione non valida: {img_name}")
    return images


def create_augmented_dataset(config):
    input_dir = config.input_dir
    output_dir = config.output_dir
    min_aug = config.min_augmented_per_image
    max_aug = config.max_augmented_per_image
    copy_original = config.copy_original
    valid_ext = config.valid_extensions
    overwrite = config.overwrite

    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    _prepare_output_dir(output_dir, class_names, overwrite)
    images = _get_images_to_process(input_dir, class_names, valid_ext)

    # Conteggi originali
    class_counts = Counter(cls for cls, _ in images)
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    # Generazione reale delle immagini
    transform = get_transforms()
    total_aug = 0

    logger.info("Inizio processamento immagini")
    logger.debug(f"Numero minimo di augmentazioni per immagine: {min_aug}")
    logger.debug(f"Numero massimo di augmentazioni per immagine: {max_aug}")
    logger.info(f"Copia immagini originali: {copy_original}")

    for cls, img_name in tqdm.tqdm(images, desc="Elaborazione immagini"):
        in_path = os.path.join(input_dir, cls, img_name)
        out_dir = os.path.join(output_dir, cls)

        try:
            if copy_original:
                shutil.copy(in_path, os.path.join(out_dir, img_name))

            image = Image.open(in_path).convert("RGB")

            # Calcola augmentazioni reali in base al bilanciamento
            orig_count = class_counts[cls]
            if orig_count == min_count:
                num_aug_cls = max_aug
            elif orig_count == max_count:
                num_aug_cls = min_aug
            else:
                # Classe intermedia: scala proporzionalmente tra min e max
                scale = (orig_count - min_count) / (max_count - min_count)
                num_aug_cls = int(round(max_aug - (max_aug - min_aug) * scale))

            for i in range(num_aug_cls):
                aug_img = transform(image)
                aug_img.save(os.path.join(out_dir, f"aug_{i}_{img_name}"))
                total_aug += 1

        except Exception as e:
            logger.warning(f"Errore con l'immagine '{in_path}': {e}")

    if copy_original:
        logger.debug(f"Copiate {len(images)} immagini originali.")
    logger.debug(f"Generate {total_aug} immagini augmentate.")
    logger.info("Augmentazione completata.")
