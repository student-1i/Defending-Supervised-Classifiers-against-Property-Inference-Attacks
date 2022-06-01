def train(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.4f}'.format(epoch, batch_idx * len(data),
                                                                        len(train_loader.dataset),
                                                                        100. * batch_idx / len(train_loader),
                                                                        loss.item()))
    return loss.item()

def test(model, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for data, target in test_loader:
            # data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().sum()
            total += data.size(0)
        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(test_loss, correct, total,
                                                                                 100. * correct.item() / total))
    return correct.item() >= total * 0.8
