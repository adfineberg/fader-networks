import time
import torch
import torch.nn as nn
import torch.optim as optim
from os.path import join
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import AttributeClassifier
from utils import split_train_val_test


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


def train_attribute_classifier():
    gpu_id = 1
    use_cuda = False
    data_dir = 'data'
    attribute_classifier_fpath = join(data_dir, 'weights', 'attr_cls.params')
    print('Loading data')
    train, valid, _ = split_train_val_test(data_dir)
    print('Creating model')
    num_attr = train.attribute_names.shape[0]
    attribute_classifier = AttributeClassifier(num_attr, use_cuda=use_cuda, gpu_id=gpu_id)
    if use_cuda:
        attribute_classifier.cuda(gpu_id)

    train_iter = DataLoader(train, batch_size=32, shuffle=True, num_workers=14)
    valid_iter = DataLoader(valid, batch_size=32, shuffle=False, num_workers=14)

    max_epochs = 90
    lr, beta1 = 2e-3, 0.5
    optimizer = optim.Adam(attribute_classifier.parameters(),
                           lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()
    print('Starting training')
    try:
        for epoch in range(1, max_epochs):
            train_model(attribute_classifier, criterion, epoch, gpu_id, optimizer, train_iter, use_cuda)
            validate_model(attribute_classifier, criterion, epoch, gpu_id, use_cuda, valid_iter)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': attribute_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, False)

    except KeyboardInterrupt:
        print('Caught Ctrl-C, interrupting training.')
    # except RuntimeError:
    #     print('RuntimeError')
    print('Saving attribute_classifier parameters to %s' % attribute_classifier_fpath)
    torch.save(attribute_classifier.state_dict(), attribute_classifier_fpath)


def validate_model(attribute_classifier, criterion, epoch, gpu_id, use_cuda, valid_iter):
    attribute_classifier.eval()
    for iteration, (x, yb, yt, _) in enumerate(valid_iter, start=1):
        if use_cuda:
            x = x.cuda(gpu_id)
            yb, yt = yb.cuda(gpu_id), yt.cuda(gpu_id)
        x, yb, yt = Variable(x), Variable(yb), Variable(yt)
        y_hat = attribute_classifier(x)

        # plot_samples(x, x_hat, prefix='valid_%d_%d' % (
        #    epoch, iteration))

        valid_loss = criterion(y_hat, yt)
        print(' Valid epoch %d, iter %d' % (
            epoch, iteration))
        print('  loss = %.6f' % (valid_loss.data[0]))


def train_model(attribute_classifier, criterion, epoch, gpu_id, optimizer, train_iter, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    attribute_classifier.train()

    end = time.time()
    for iteration, (x, yb, yt, _) in enumerate(train_iter):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            x = x.cuda(gpu_id)
            yb, yt = yb.cuda(gpu_id), yt.cuda(gpu_id)
        x, yb, yt = Variable(x), Variable(yb), Variable(yt)
        # print yb.data.cpu().numpy().shape
        # print yt.data.cpu().numpy().shape
        optimizer.zero_grad()
        y_hat = attribute_classifier(x)

        loss = criterion(y_hat, yt)
        loss.backward()
        optimizer.step()

        losses.update(loss.data[0], x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, iteration, len(train_iter), batch_time=batch_time, loss=losses))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    train_attribute_classifier()

