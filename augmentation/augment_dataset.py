import os
import shutil
import sys
import time
from collections import Counter
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from logger import get_logger
from transform_factory import get_transforms

logger = get_logger()


# --- Funzioni di utilità ---
def _ensure_dirs(dirs):
    """Crea le cartelle se non esistono."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _scan_images(input_dir, valid_ext):
    """Restituisce lista di tuple (cls, img_name) e lista classi presenti."""
    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
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
    return images, class_names


# --- Preparazione directory ---
def _prepare_output_dir(output_dir, class_names, overwrite, analysis_dirs):
    """Crea cartelle di output e cancella quelle esistenti se overwrite=True."""
    if overwrite:
        if os.path.exists(output_dir):
            logger.info(f"Overwrite abilitato: elimino la cartella esistente '{output_dir}'")
            shutil.rmtree(output_dir)
        for dir_path in analysis_dirs:
            if os.path.exists(dir_path):
                logger.info(f"Cancellazione cartella di report esistente '{dir_path}'")
                shutil.rmtree(dir_path)
    else:
        if os.path.exists(output_dir):
            logger.error(f"La cartella '{output_dir}' esiste già. Imposta overwrite=true per sovrascrivere.")
            sys.exit(1)

    # Creazione cartelle nuove
    _ensure_dirs([output_dir] + [os.path.join(output_dir, cls) for cls in class_names] + analysis_dirs)


# --- Funzione di augmentazione ---
def _create_augmented_dataset(input_dir, output_dir, min_aug, max_aug, copy_original, valid_extensions):
    images, class_names = _scan_images(input_dir, valid_extensions)

    # Conteggi classi per bilanciamento
    class_counts = Counter(cls for cls, _ in images)
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    transform = get_transforms()
    total_aug = 0

    logger.info("Inizio processamento immagini")
    logger.info(f"Copia immagini originali: {copy_original}")

    for cls, img_name in tqdm.tqdm(images, desc="Elaborazione immagini"):
        in_path = os.path.join(input_dir, cls, img_name)
        out_dir = os.path.join(output_dir, cls)

        try:
            if copy_original:
                shutil.copy(in_path, os.path.join(out_dir, img_name))

            image = Image.open(in_path).convert("RGB")

            # Calcolo numero di augmentazioni basato sul bilanciamento
            orig_count = class_counts[cls]
            if orig_count == min_count:
                num_aug_cls = max_aug
            elif orig_count == max_count:
                num_aug_cls = min_aug
            else:
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


# --- Funzione di analisi ---
def _analyze_dataset(input_dir, valid_extensions, show_chart, output_dir):
    _ensure_dirs([output_dir])
    images, _ = _scan_images(input_dir, valid_extensions)
    class_counts = Counter(cls for cls, _ in images)

    # Log conteggi classi
    for cls, count in class_counts.items():
        logger.info(f"Classe '{cls}': {count} immagini")

    # Grafico distribuzione classi
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.title("Distribuzione immagini per classe")
    plt.ylabel("Numero immagini")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"class_distribution_{timestamp}.png"))
    if show_chart:
        plt.show()
    plt.close()


# --- Funzione principale ---
def run_full_augmentation(config):
    input_dir = config.input_dir
    output_dir = config.output_dir
    analysis_dir = config.analisys_dir
    show_chart = config.show_chart
    valid_extensions = config.valid_extensions
    overwrite = config.overwrite

    min_aug = config.min_augmented_per_image
    max_aug = config.max_augmented_per_image
    copy_original = config.copy_original

    # Lista classi presenti
    _, class_names = _scan_images(input_dir, valid_extensions)

    # Preparazione directory
    _prepare_output_dir(output_dir, class_names, overwrite, [os.path.join(analysis_dir, "before"),
                                                             os.path.join(analysis_dir, "after")])

    # Analisi prima dell'augmentazione
    _analyze_dataset(input_dir, valid_extensions, show_chart, output_dir=os.path.join(analysis_dir, "before"))

    # Augmentazione
    _create_augmented_dataset(input_dir, output_dir, min_aug, max_aug, copy_original, valid_extensions)

    # Analisi dopo l'augmentazione
    _analyze_dataset(output_dir, valid_extensions, show_chart, output_dir=os.path.join(analysis_dir, "after"))
