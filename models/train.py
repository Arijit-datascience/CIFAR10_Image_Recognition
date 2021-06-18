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
    l1_factor = 0.0001
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Calculate loss
        loss = F.nll_loss(output, target)
        L1_loss = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        reg_loss = 0 
        for param in model.parameters():
            zero_vector = torch.rand_like(param) * 0
            reg_loss += L1_loss(param,zero_vector)
        loss += l1_factor * reg_loss

        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        train_acc.append(100*correct/processed)

    print('\nTrain set: Average loss: {:.4f}, Train Accuracy: {}/{} ({:.2f}%)\n'.format(
    loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset)))
