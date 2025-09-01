# Garbage Classification Project

## Descrizione
Questo progetto implementa un sistema di classificazione dei rifiuti utilizzando reti neurali convoluzionali (CNN) come EfficientNet-B0 e ResNet-18. Il progetto include due componenti principali: l'**augmentation** del dataset e l'**addestramento/test** dei modelli.

## Struttura del progetto
- `root/data/garbage_classification/` → Dataset originale  
- `root/data/garbage_augmented/` → Dataset generato dall'augmentation  
- `root/data/analysis/` → Distribuzione immagini prima e dopo augmentation  
- `root/saved_models/` → Modelli salvati
- `root/logs/` → Log del terminale  
- `root/runs/` → TensorBoard logs  
- `root/config/` → Config per addestramento  
- `root/augmentation/config/` → Config per augmentation  
- `root/main.py` → Script principale di addestramento  
- `root/augmentation/run_augmentation.py` → Script per augmentation  


## Salvataggio e test dei modelli
- I modelli migliori vengono salvati in `saved_models/` solo se mostrano miglioramenti significativi durante l'addestramento.
- Modelli finali salvati nello stesso percorso
- Per testare un modello già addestrato, impostare nel config `load_model: true`

## Logging
- Log del terminale: `logs/`
- TensorBoard logs: `runs/`

## Configurazioni
- **Addestramento:** `config/config.json`
- **Augmentation:** `augmentation/config/config.json`

## Requisiti
Tutti i requisiti software e librerie sono riportati in `environment.yaml`.

## Output attesi
- Dopo l'augmentation:
  - Dataset augmentato (`garbage_augmented`)
  - Analisi distribuzione immagini (`analysis`)
- Dopo l'addestramento:
  - Modelli salvati (`saved_models/`)
  - Log (`logs/`)
  - TensorBoard logs (`runs/`)
