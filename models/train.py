import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torch.optim.lr_scheduler import StepLR

def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
    
    # Train network
    model.train()
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Calculate loss
        loss = F.nll_loss(output, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        print('\nTrain set: Average loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)\n'.format(
        	loss, correct, len(train_loader.dataset),
        	100. * correct / len(train_loader.dataset)))

        train_acc.append(100*correct/processed)