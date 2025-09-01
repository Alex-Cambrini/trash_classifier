# Garbage Classification Project

## Descrizione
Questo progetto implementa un sistema di classificazione dei rifiuti utilizzando reti neurali convoluzionali (CNN) come EfficientNet-B0 e ResNet-18. Il progetto include due componenti principali: l'**augmentation** del dataset e l'**addestramento/test** dei modelli.

## Setup ambiente

Creare l'ambiente Conda dal file `environment.yaml`:

```bash
conda env create -f environment.yaml
```
Attivare l'ambiente:
```bash
conda activate garbage_classification
```
## Comandi principali
Eseguire l’augmentation:
```bash
python ./augmentation/run_augmentation.py
```
Addestramento/Test dei modelli:
```bash
python main.py
```

## Struttura del progetto
- `data/garbage_classification/` → Dataset originale  
- `data/garbage_augmented/` → Dataset generato dall'augmentation  
- `data/analysis/` → Distribuzione immagini prima e dopo augmentation  
- `saved_models/` → Modelli migliori vengono salvati
- `logs/` → Log del terminale  
- `runs/` → TensorBoard logs  
- `config/` → Config per addestramento  
- `augmentation/config/` → Config per augmentation  
- `main.py` → Script principale di addestramento  
- `augmentation/run_augmentation.py` → Script per augmentation  

## Output attesi
- Dataset augmentato (`data/garbage_augmented/`)  
- Analisi distribuzione immagini (`data/analysis/`)  
- Modelli salvati (`saved_models/`)  
- Log (`logs/`)  
- TensorBoard logs (`runs/`)  

