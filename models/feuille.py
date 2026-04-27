import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image, ImageEnhance, ImageFilter
from torch.amp import autocast, GradScaler
import warnings

warnings.filterwarnings("ignore")


torch.backends.cudnn.benchmark = True

from config import (
    FEUILLE_TRAIN_DIR,
    FEUILLE_VALID_DIR,
    FEUILLE_MODEL_PATH,
    FEUILLE_CLASSES_PATH,
    FEUILLE_BATCH_SIZE,
    FEUILLE_EPOCHS,
    FEUILLE_LR,
    FEUILLE_IMG_SIZE,
    FEUILLE_DEVICE,
)

DEVICE = FEUILLE_DEVICE
BATCH_SIZE = FEUILLE_BATCH_SIZE
EPOCHS = FEUILLE_EPOCHS
LR = FEUILLE_LR
IMG_SIZE = FEUILLE_IMG_SIZE


def get_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            # Simule flou de bougé / mise au point ratée (30% du temps)
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),
            # Simule basse résolution de téléphone
            transforms.RandomApply(
                [
                    transforms.Resize((120, 120)),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                ],
                p=0.1,
            ),
            transforms.ToTensor(),
            # Normlaisation avec les stats de ImageNet (moyenne et écart type)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    valid_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, valid_tf


def preprocess_phone_photo(img: Image.Image) -> Image.Image:
    # Améliore la netteté
    img = img.filter(ImageFilter.SHARPEN)
    # Booste légèrement le contraste
    img = ImageEnhance.Contrast(img).enhance(1.1)
    # Booste légèrement la saturation pour compenser les photos ternes
    img = ImageEnhance.Color(img).enhance(1.05)
    return img


def build_model(num_classes: int) -> nn.Module:
    # Charge EfficientNetB0 pour transfert learning
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Geler le backbone, entraîner seulement le classifier car il sait déja reconnaitre des formes générales
    for param in model.features.parameters():
        param.requires_grad = False
    # Get taille d'entrée du classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(
            p=0.3
        ),  # drop 30% des neurones pendant l'entrainement pour évier sur entrainement
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE).to(memory_format=torch.channels_last)


def train_model_feuille():
    print(f"[feuille] Device : {DEVICE}")

    train_tf, valid_tf = get_transforms()
    # Load img
    train_ds = datasets.ImageFolder(FEUILLE_TRAIN_DIR, transform=train_tf)
    valid_ds = datasets.ImageFolder(FEUILLE_VALID_DIR, transform=valid_tf)
    # Découpe données en mini batch + shuffle pour apprentissage
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    num_classes = len(train_ds.classes)
    # print(num_classes)
    print(
        f"[feuille] {num_classes} classes  {len(train_ds)} images train / {len(valid_ds)} images valid"
    )

    model = build_model(num_classes)
    # Get loss
    criterion = nn.CrossEntropyLoss()
    # Ajuster poids classifier (correcteur)
    optimizer = Adam(model.classifier.parameters(), lr=LR)
    # Stabilise apprentissage haut -> bas -> haut (cos)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    patience = 7
    counter = 0
    scaler = GradScaler(DEVICE.type)
    FINE_TUNE_EPOCH = EPOCHS // 2 + 1

    for epoch in range(1, EPOCHS + 1):
        # Phase 2 : fine-tuning à mi-parcours
        if epoch == FINE_TUNE_EPOCH:
            print("[feuille] Fine-tuning : dégel des 5 dernières couches du backbone")
            for param in model.features[-5:].parameters():
                param.requires_grad = True
            optimizer = Adam(
                [
                    {"params": model.classifier.parameters(), "lr": LR},
                    {"params": model.features[-5:].parameters(), "lr": LR / 10},
                ]
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=10)
            counter = 0
        # Entraînement
        model.train()
        train_loss, train_correct = 0.0, 0
        # Loop on batch
        for images, labels in train_loader:
            images = images.to(DEVICE, memory_format=torch.channels_last)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            # Make a prediction
            with autocast(device_type=DEVICE.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            # Compte combien d'images sont bien classés dans ce batch
            train_correct += (outputs.argmax(1) == labels).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        # Inutile de stocker les gradients car on apprend pas
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(DEVICE, memory_format=torch.channels_last)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                valid_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_ds) * 100
        valid_acc = valid_correct / len(valid_ds) * 100
        print(
            f"  Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss/len(train_ds):.4f}  train_acc={train_acc:.1f}%  "
            f"valid_loss={valid_loss/len(valid_ds):.4f}  valid_acc={valid_acc:.1f}%"
        )

        # Sauvegarde meilleur modèle + early stopping
        if valid_acc > best_acc:
            best_acc = valid_acc
            counter = 0
            os.makedirs(os.path.dirname(FEUILLE_MODEL_PATH), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "best_acc": best_acc,
                },
                FEUILLE_MODEL_PATH,
            )
            print(f"  Meilleur modèle sauvegardé (valid_acc={best_acc:.1f}%)")
        else:
            counter += 1
            print(f"  Pas d'amélioration ({counter}/{patience})")
            if counter >= patience:
                print(f"  Early stopping à l'epoch {epoch}")
                break

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
        checkpoint = torch.load(
            FEUILLE_MODEL_PATH, map_location=DEVICE, weights_only=True
        )
        model = build_model(len(classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, classes
    return train_model_feuille()[:2]


if __name__ == "__main__":
    train_model_feuille()
