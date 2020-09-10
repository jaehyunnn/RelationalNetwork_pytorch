from __future__ import print_function, division

from time import time

def train(epoch,model,loss_fn,optimizer,dataloader,use_cuda=True,log_interval=50,):
    model.train()
    train_loss = 0
    start_time = time()
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_img, input_qst, label = batch['image'], batch['question'], batch['answer']
        if use_cuda:
            input_img = input_img.cuda()
            input_qst = input_qst.cuda()
            label = label.cuda()
        output = model(input_img, input_qst)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            pred = output.data.max(1)[1]
            correct = pred.eq(label.data).cpu().sum()
            accuracy = correct * 100. / len(label)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f} \t\tAccuracy: {:.2f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss.data, accuracy))

    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f} --- {:.2f}s'.format(train_loss, (time()-start_time)))
    return train_loss

def test(model,dataloader,n_data,use_cuda=True):
    model.eval()
    total_correct = 0
    for batch_idx, batch in enumerate(dataloader):
        input_img, input_qst, label = batch['image'], batch['question'], batch['answer']
        if use_cuda:
            input_img = input_img.cuda()
            input_qst = input_qst.cuda()
            label = label.cuda()
        output = model(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        total_correct += correct
    test_acc = total_correct * 100. / n_data
    print('Test set: Accuracy: {:.2f}\n'.format(test_acc))
    return test_acc
