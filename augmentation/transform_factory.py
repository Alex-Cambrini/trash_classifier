from torchvision import transforms

def get_transforms():
    """
    Restituisce la pipeline di trasformazioni per l'augmentation.
    Puoi personalizzare qui le trasformazioni secondo necessità.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
