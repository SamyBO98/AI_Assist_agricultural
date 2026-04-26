import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings("ignore")
 
from config import (
    FEUILLE_TRAIN_DIR, FEUILLE_VALID_DIR,
    FEUILLE_MODEL_PATH, FEUILLE_CLASSES_PATH,
    FEUILLE_BATCH_SIZE, FEUILLE_EPOCHS,
    FEUILLE_LR, FEUILLE_IMG_SIZE, FEUILLE_DEVICE,
)
 
DEVICE     = FEUILLE_DEVICE
BATCH_SIZE = FEUILLE_BATCH_SIZE
EPOCHS     = FEUILLE_EPOCHS
LR         = FEUILLE_LR
IMG_SIZE   = FEUILLE_IMG_SIZE
 
 
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        #Normlaisation avec les stats de ImageNet (moyenne et écart type)
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    valid_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, valid_tf
 
 
def build_model(num_classes: int) -> nn.Module:
    #Charge EfficientNetB0 pour transfert learning
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Geler le backbone, entraîner seulement le classifier car il sait déja reconnaitre des formes générales
    for param in model.features.parameters():
        param.requires_grad = False
    #Get taille d'entrée du classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3), #drop 30% des neurones pendant l'entrainement pour évier sur entrainement
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)
 
 
def train_model_feuille():
    print(f"[feuille] Device : {DEVICE}")
    
    train_tf, valid_tf = get_transforms()
    #Load img
    train_ds = datasets.ImageFolder(FEUILLE_TRAIN_DIR, transform=train_tf)
    valid_ds = datasets.ImageFolder(FEUILLE_VALID_DIR, transform=valid_tf)
    #Découpe données en mini batch + shuffle pour apprentissage
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = len(train_ds.classes)
    #print(num_classes)
    print(f"[feuille] {num_classes} classes  {len(train_ds)} images train / {len(valid_ds)} images valid")
 
    model = build_model(num_classes)
    #Get loss
    criterion = nn.CrossEntropyLoss()
    #Ajuster poids classifier (correcteur)
    optimizer = Adam(model.classifier.parameters(), lr=LR)
    #Stabilise apprentissage haut -> bas -> haut (cos)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        #Entraînement
        model.train()
        train_loss, train_correct = 0.0, 0
        #Loop on batch
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            #Make a prediction
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            #Compte combien d'images sont bien classés dans ce batch
            train_correct += (outputs.argmax(1) == labels).sum().item()
 
        scheduler.step()
 
        #Validation
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        #Inutile de stocker les gradients car on apprend pas 
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                valid_correct += (outputs.argmax(1) == labels).sum().item()
 
        train_acc = train_correct / len(train_ds) * 100
        valid_acc = valid_correct / len(valid_ds) * 100
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss/len(train_ds):.4f}  train_acc={train_acc:.1f}%  "
              f"valid_loss={valid_loss/len(valid_ds):.4f}  valid_acc={valid_acc:.1f}%")
 
        if valid_acc > best_acc:
            best_acc = valid_acc
            os.makedirs(os.path.dirname(FEUILLE_MODEL_PATH), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "best_acc": best_acc,
            }, FEUILLE_MODEL_PATH)
            print(f"Meilleur modèle sauvegardé (valid_acc={best_acc:.1f}%)")
 
    with open(FEUILLE_CLASSES_PATH, "w") as f:
        json.dump(train_ds.classes, f, ensure_ascii=False)
    print(f"[feuille] Classes sauvegardées → {FEUILLE_CLASSES_PATH}")
    print(f"[feuille] Entraînement terminé meilleure valid_acc={best_acc:.1f}%")
 
    return model, train_ds.classes, best_acc
 
 
def load_or_train_feuille():
    if os.path.exists(FEUILLE_MODEL_PATH) and os.path.exists(FEUILLE_CLASSES_PATH):
        print(f"[feuille] Chargement depuis {FEUILLE_MODEL_PATH}")
        with open(FEUILLE_CLASSES_PATH) as f:
            classes = json.load(f)
        checkpoint = torch.load(FEUILLE_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model = build_model(len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, classes
    return train_model_feuille()[:2]
 
 
if __name__ == "__main__":
    train_model_feuille()