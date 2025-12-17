import faulthandler
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import timm
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import fine_tune


def load_datasets(dataset_dir: str) -> tuple:
    train_dir = f'{dataset_dir}/train'
    val_dir = f'{dataset_dir}/val'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    return train_dataset, val_dataset


def load_loaders(train_dataset, val_dataset) -> tuple:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def load_raw_resnet_model(num_classes: int) -> tuple:
    resnet50 = torchvision_models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

    return resnet50, ['fc', 'layer4']


def load_raw_efficientnet_model(num_classes: int) -> tuple:
    efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
    efficientnet.classifier = nn.Linear(efficientnet.classifier.in_features, num_classes)

    return efficientnet, ['classifier', 'blocks.6']


def train_or_load_model(model_name: str, load_raw_model_funcs, train_loader, val_loader, device, trained_models_path: str, force_retrain: bool = False):
    trained_model_path_wo_extension = f'{trained_models_path}/{model_name}'
    trained_model_path = f'{trained_model_path_wo_extension}.pth'
    trained_model_history_path = f'{trained_model_path_wo_extension}.pkl'

    if not force_retrain and os.path.exists(trained_model_path):
        print(f'Загрузка модели {model_name}')
        model = torch.load(trained_model_path, weights_only=False, map_location=device)
        history = None
        if os.path.exists(trained_model_history_path):
            with open(trained_model_history_path, 'rb') as f:
                history = pickle.load(f)
        if history is None:
            print(f'Не удалось загрузить историю обучения модели {model_name}')
    else:
        print(f'Обучение модели {model_name}')
        model, history = fine_tune.fine_tune_model(model_name, load_raw_model_funcs, train_loader, val_loader, device)

        torch.save(model, trained_model_path)
        with open(trained_model_history_path, 'wb') as f:
            pickle.dump(history, f)

    model.eval()
    return model_name, model, history


def show_metrics(model, data_loader, classes, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            prediction = torch.argmax(outputs, dim=1)
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_predictions, target_names=classes, digits=4)
    cm = confusion_matrix(all_labels, all_predictions)

    print("Отчет классификации:\n", report)
    print("Матрица ошибок:\n", cm)

    return report, cm


def visualize_metrics(history: dict, model_name: str):
    epochs = range(1, len(history['train_loss']) + 1)

    # График loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'b-', label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # График accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'r-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'b-', label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix(cm, classes: list, model_name: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Матрица ошибок модели {model_name}')
    plt.xlabel('Предсказано')
    plt.ylabel('Факт')
    plt.show()


def show_misclassified(model, val_loader, device, classes: list, max_images=5):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            mismatches = (predictions != labels).nonzero(as_tuple=False)

            for idx in mismatches:
                img = inputs[idx][0].cpu().numpy().transpose((1, 2, 0))
                true_label = classes[labels[idx].item()]
                prediction_label = classes[predictions[idx].item()]
                misclassified.append((img, true_label, prediction_label))
                if len(misclassified) >= max_images:
                    break
            if len(misclassified) >= max_images:
                break

    plt.figure(figsize=(12, 6))
    for i, (img, true_lbl, prediction_label) in enumerate(misclassified):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        plt.title(f"Факт: {true_lbl}\nПредсказание: {prediction_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


DATASET_DIR = '../1/result'
TRAINED_MODELS_DIR = 'trained_models'
FORCE_RETRAIN = True


def main():
    global DATASET_DIR, TRAINED_MODELS_DIR, FORCE_RETRAIN

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Запуск на {device_str}\n')
    device = torch.device(device_str)

    train_dataset, val_dataset = load_datasets(DATASET_DIR)
    train_loader, val_loader = load_loaders(train_dataset, val_dataset)
    classes = train_dataset.classes
    num_classes = len(classes)

    model_names = [
        'resnet50',
        'efficientnet_b0',
    ]

    load_raw_model_funcs = dict.fromkeys(model_names)
    load_raw_model_funcs[model_names[0]] = lambda: load_raw_resnet_model(num_classes)
    load_raw_model_funcs[model_names[1]] = lambda: load_raw_efficientnet_model(num_classes)

    models = []
    for model_name in model_names:
        model = train_or_load_model(model_name, load_raw_model_funcs, train_loader, val_loader, device, TRAINED_MODELS_DIR, FORCE_RETRAIN)
        models.append(model)
        print()

    for model_name, model, history in models:
        print(f'Сбор и визуализация метрик модели {model_name}')

        report, cm = show_metrics(model, val_loader, classes, device)
        if history is not None:
            visualize_metrics(history, model_name)
        visualize_confusion_matrix(cm, classes, model_name)
        show_misclassified(model, val_loader, device, classes)
        print()


if __name__ == '__main__':
    faulthandler.enable()
    main()
