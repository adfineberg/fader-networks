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

    train, valid, _ = split_train_val_test(data_dir)

    num_attr = train.attribute_names.shape[0]
    attribute_classifier = AttributeClassifier(num_attr, use_cuda=use_cuda, gpu_id=gpu_id)
    if use_cuda:
        attribute_classifier.cuda(gpu_id)

    train_iter = DataLoader(train, batch_size=32, shuffle=True, num_workers=8)
    valid_iter = DataLoader(valid, batch_size=32, shuffle=False, num_workers=8)

    max_epochs = 1000
    lr, beta1 = 2e-3, 0.5
    optimizer = optim.Adam(attribute_classifier.parameters(),
                           lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    num_iters = 0

    try:
        for epoch in range(1, max_epochs):
            attribute_classifier.train()
            for iteration, (x, yb, yt, _) in enumerate(train_iter, start=1):
                if use_cuda:
                    x = x.cuda(gpu_id)
                    yb, yt = yb.cuda(gpu_id), yt.cuda(gpu_id)
                x, yb, yt = Variable(x), Variable(yb), Variable(yt)
                #print yb.data.cpu().numpy().shape
                #print yt.data.cpu().numpy().shape
                optimizer.zero_grad()
                y_hat = attribute_classifier(x)

                #if (epoch == 1) or (epoch % sample_every == 0):
                #if (epoch % sample_every == 0):
                #    plot_samples(x, x_hat, prefix='train_%d_%d' % (
                #        epoch, iteration))

                # send the output of the encoder as a new Variable that is not
                # part of the backward pass
                # not sure if this is the correct way to do so
                # https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/9

                loss = criterion(y_hat, yt)
                loss.backward()
                optimizer.step()

                print(' Train epoch %d, iter %d' % (
                    epoch, iteration))
                print('  loss = %.6f' % (loss.data[0]))

                num_iters += 1
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': best_prec1,
            #     'optimizer' : optimizer.state_dict(),
            # }, is_best)
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

    except KeyboardInterrupt:
        print('Caught Ctrl-C, interrupting training.')
    except RuntimeError:
        print('RuntimeError')
    print('Saving encoder/decoder parameters to %s' % attribute_classifier_fpath)
    torch.save(attribute_classifier.state_dict(), attribute_classifier_fpath)


if __name__ == '__main__':
    train_attribute_classifier()

