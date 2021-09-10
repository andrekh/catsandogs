import torch


def train(model, max_epochs, optimizer, training_generator, validation_generator, device, neptune_run):
    loss_fn = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0
    for epoch in range(max_epochs):
        model.train()
        loss_train = 0.0
        for imgs, labels in training_generator:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        print(f"Epoch {epoch + 1}\t\t Training loss: {loss_train / len(training_generator)}")

        neptune_run['train/loss'].log(loss_train / len(training_generator))

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for imgs, labels in validation_generator:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Epoch {epoch + 1}\t\t val accuracy: {correct / total}")
            neptune_run['train/val_acc'].log(correct / total)

        if epoch == 0:
            best_val_acc = correct / total
        else:
            best_val_acc = max(best_val_acc, correct / total)
        neptune_run['best_val_acc'] = best_val_acc
