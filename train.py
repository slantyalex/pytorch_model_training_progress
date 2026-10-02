import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from animalvision import (
    Config,
    get_dataloaders,
    AnimalVisionModel,
    compute_metrics,
    plot_training_curves
)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics["accuracy"]


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics["accuracy"]


def main():
    print(f"[+] Using device: {Config.DEVICE}")
    train_loader, val_loader, _ = get_dataloaders()

    model = AnimalVisionModel(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()

    if hasattr(model, 'model') and hasattr(model.model, 'fc'):
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.model.named_parameters() if "fc" not in n], 'lr': 1e-4},
            {'params': model.model.fc.parameters(), 'lr': Config.LEARNING_RATE}
        ], weight_decay=1e-2)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(Config.EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        v_loss, v_acc = validate(model, val_loader, criterion, Config.DEVICE)

        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        current_lr = optimizer.param_groups[-1]['lr']
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] (LR: {current_lr:.6f}) -> "
              f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, Config.MODEL_SAVE_PATH)
            print(f"    [--> Saved Best Checkpoint to {Config.MODEL_SAVE_PATH}]")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()