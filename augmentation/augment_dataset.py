import os
import shutil
import sys
from collections import Counter
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
from transform_factory import get_transforms

class AugmentationRunner:
    def __init__(self, config, logger, run_name):
        self.config = config
        self.logger = logger
        self.run_name = run_name

    # --- Funzioni di utilità ---
    def _ensure_dirs(self, dirs):
        """Crea le cartelle se non esistono."""
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def _scan_images(self, input_dir, valid_ext):
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
                        self.logger.warning(f"Immagine corrotta o non apribile: {img_path}")
        return images, class_names

    # --- Preparazione directory ---
    def _prepare_output_dir(self, output_dir, class_names, overwrite, analysis_dirs):
        """Crea cartelle di output e cancella quelle esistenti se overwrite=True."""
        if overwrite:
            if os.path.exists(output_dir):
                self.logger.info(f"Overwrite abilitato: elimino la cartella esistente '{output_dir}'")
                shutil.rmtree(output_dir)
            for dir_path in analysis_dirs:
                if os.path.exists(dir_path):
                    self.logger.debug(f"Cancellazione cartella di report esistente '{dir_path}'")
                    shutil.rmtree(dir_path)
        else:
            if os.path.exists(output_dir):
                self.logger.error(f"La cartella '{output_dir}' esiste già. Imposta overwrite=true per sovrascrivere.")
                sys.exit(1)

        self._ensure_dirs([output_dir] + [os.path.join(output_dir, cls) for cls in class_names] + analysis_dirs)

    # --- Funzione di augmentazione ---
    def _create_augmented_dataset(self, input_dir, output_dir, min_aug, max_aug, copy_original, valid_extensions):
        images, _ = self._scan_images(input_dir, valid_extensions)

        class_counts = Counter(cls for cls, _ in images)
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())

        transform = get_transforms()
        total_aug = 0

        self.logger.info("Inizio processamento immagini")
        self.logger.info(f"Copia immagini originali: {copy_original}")

        for cls, img_name in tqdm.tqdm(images, desc="Elaborazione immagini"):
            in_path = os.path.join(input_dir, cls, img_name)
            out_dir = os.path.join(output_dir, cls)
            try:
                if copy_original:
                    shutil.copy(in_path, os.path.join(out_dir, img_name))

                image = Image.open(in_path).convert("RGB")

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
                self.logger.warning(f"Errore con l'immagine '{in_path}': {e}")

        if copy_original:
            self.logger.debug(f"Copiate {len(images)} immagini originali.")
        self.logger.debug(f"Generate {total_aug} immagini augmentate.")
        self.logger.info("Augmentazione completata.")

    # --- Funzione di analisi ---
    def _analyze_dataset(self, input_dir, valid_extensions, show_chart, output_dir):
        self._ensure_dirs([output_dir])
        images, _ = self._scan_images(input_dir, valid_extensions)
        class_counts = Counter(cls for cls, _ in images)

        for cls, count in class_counts.items():
            self.logger.info(f"Classe '{cls}': {count} immagini")

        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
        plt.title("Distribuzione immagini per classe")
        plt.ylabel("Numero immagini")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_distribution_{self.run_name}.png"))
        if show_chart:
            plt.show()
        plt.close()

    # --- Funzione principale ---
    def run(self):
        input_dir = self.config.input_dir
        output_dir = self.config.output_dir
        analysis_dir = self.config.analisys_dir
        show_chart = self.config.show_chart
        valid_extensions = self.config.valid_extensions
        overwrite = self.config.overwrite

        min_aug = self.config.min_augmented_per_image
        max_aug = self.config.max_augmented_per_image
        copy_original = self.config.copy_original

        self.logger.debug(f"Show chart: {show_chart} (i grafici verranno mostrati solo se TRUE)")

        _, class_names = self._scan_images(input_dir, valid_extensions)

        self._prepare_output_dir(output_dir, class_names, overwrite,
                                 [os.path.join(analysis_dir, "before"),
                                  os.path.join(analysis_dir, "after")])

        self._analyze_dataset(input_dir, valid_extensions, show_chart,
                              output_dir=os.path.join(analysis_dir, "before"))

        self._create_augmented_dataset(input_dir, output_dir, min_aug, max_aug, copy_original, valid_extensions)

        self._analyze_dataset(output_dir, valid_extensions, show_chart,
                              output_dir=os.path.join(analysis_dir, "after"))
