import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def freeze_layers_except(model, layers_to_except):
    for name, param in model.named_parameters():
        param.requires_grad = False
        for layer_to_except in layers_to_except:
            if str(layer_to_except) in name:
                param.requires_grad = True


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_predictions)
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs: int, history: dict):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        epoch_duration = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{num_epochs} | Duration: {epoch_duration:.2f} sec | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


def fine_tune_model(model_name, load_raw_model_funcs, train_loader, val_loader, device, num_epochs_head: int = 5, num_epochs_ft: int = 5):
    model, fc_layers = load_raw_model_funcs[model_name]()
    model = model.to(device)

    freeze_layers_except(model, fc_layers[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print('Обучение классифицирующего слоя')
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs_head, history)
    print()

    freeze_layers_except(model, fc_layers)
    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    print('Дообучение остальных слоёв')
    train_model(model, train_loader, val_loader, criterion, optimizer_ft, device, num_epochs_ft, history)
    print()

    return model, history

