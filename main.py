import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import network
import dataset
from auxiliary.transforms import batch2gif

from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

from colorama import Fore, Style

Style.RESET_ALL

"""=========================INPUT ARGUMENTS====================="""

parser = argparse.ArgumentParser()

parser.add_argument('--split',        default=-1,   type=int, help='Train/test classes split. Use -1 for kinetics2ucf')
parser.add_argument('--dataset',      default='ucf101',   type=str, help='Dataset: [kinetics2oboth, kinetics2others, sun2both]')

parser.add_argument('--train_samples',  default=-1,  type=int, help='Reduce number of train samples to the given value')
parser.add_argument('--class_total',  default=-1,  type=int, help='For debugging only. Reduce the total number of classes')

parser.add_argument('--clip_len',     default=16,   type=int, help='Number of frames of each sample clip')
parser.add_argument('--n_clips',     default=1,   type=int, help='Number of clips per video')

parser.add_argument('--class_overlap', default=0.05,  type=float, help='tau. see Eq.3 in main paper')

### General Training Parameters
parser.add_argument('--lr',           default=1e-3, type=float, help='Learning Rate for network parameters.')
parser.add_argument('--n_epochs',     default=150,   type=int,   help='Number of training epochs.')
parser.add_argument('--bs',           default=22,   type=int,   help='Mini-Batchsize size per GPU.')
parser.add_argument('--size',         default=112,  type=int,   help='Image size in input.')

parser.add_argument('--fixconvs', action='store_true', default=False,   help='Freezing conv layers')
parser.add_argument('--nopretrained', action='store_false', default=True,   help='Pretrain network.')

##### Network parameters
parser.add_argument('--network', default='r2plus1d_18', type=str,
                    help='Network backend choice: [resnet18, r2plus1d_18, r3d_18, c3d].')

### Paths to datasets and storage folder
parser.add_argument('--save_path',    default='/workplace/debug/', type=str, help='Where to save log and checkpoint.')
parser.add_argument('--weights',      default=None, type=str, help='Weights to load from a previously run.')
parser.add_argument('--progressbar', action='store_true', default=False,   help='Show progress bar during train/test.')
parser.add_argument('--evaluate', action='store_true', default=False,   help='Evaluation only using 25 clips per video')

##### Read in parameters
opt = parser.parse_args()

opt.multiple_clips = False
opt.kernels = multiprocessing.cpu_count()

"""=================================DATALOADER SETUPS====================="""
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    opt.bs = opt.bs * torch.cuda.device_count()

print('Total batch size: %d' % opt.bs)
dataloaders = dataset.get_dataloaders(opt)
if not opt.evaluate:
    opt.n_classes = dataloaders['training'][0].dataset.class_embed.shape[0]
else:
    opt.n_classes = dataloaders['testing'][0].dataset.class_embed.shape[0]

"""=================================OUTPUT FOLDER====================="""
opt.savename = opt.save_path + '/'
if not opt.evaluate:
    opt.savename += '%s/CLIP%d_LR%f_%s_BS%d' % (
            opt.dataset, opt.clip_len,
            opt.lr, opt.network, opt.bs)

    if opt.class_overlap > 0:
        opt.savename += '_CLASSOVERLAP%.2f' % opt.class_overlap

    if opt.class_total != -1:
        opt.savename += '_NCLASS%d' % opt.class_total

    if opt.train_samples != -1:
        opt.savename += '_NTRAIN%d' % opt.train_samples

    if opt.fixconvs:
        opt.savename += '_FixedConvs'

    if not opt.nopretrained:
        opt.savename += '_NotPretrained'

    count = 1
    while os.path.exists(opt.savename):
        opt.savename += '_{}'.format(count)
        count += 1

    if opt.split != -1:
        opt.savename += '/split%d' % opt.split

else:
    opt.weights = opt.savename + 'checkpoint.pth.tar'
    opt.savename += '/evaluation/'


if not os.path.exists(opt.savename+'/samples/'):
    os.makedirs(opt.savename+'/samples/')

"""=============================NETWORK SETUP==============================="""
opt.device = torch.device('cuda')
model      = network.get_network(opt)

if opt.weights and opt.weights != "none":
    #model.load_state_dict(torch.load(opt.weights)['state_dict'])
    j = len('module.')
    weights = torch.load(opt.weights)['state_dict']
    model_dict = model.state_dict()
    weights = {k[j:]: v for k, v in weights.items() if k[j:] in model_dict.keys()}
    # if not opt.evaluate:
    #     weights = {k: v for k, v in weights.items() if 'regressor' not in k}
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    print("LOADED MODEL:  ", opt.weights)

model = nn.DataParallel(model)
_ = model.to(opt.device)

"""==========================OPTIM SETUP=================================="""
criterion = torch.nn.MSELoss().to(opt.device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
if opt.lr == 1e-3:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.1)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.8*opt.n_epochs)], gamma=0.1)


"""===========================TRAINER FUNCTION==============================="""


def train_one_epoch(train_dataloader, model, optimizer, criterion, opt, epoch):
    """
    This function is called every epoch to perform training of the network over one full
    (randomized) iteration of the dataset.
    """
    class_embedding = train_dataloader.dataset.class_embed
    class_names = train_dataloader.dataset.class_name
    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader
    if opt.progressbar:
        data_iterator = tqdm(train_dataloader, desc='Epoch {} Training...'.format(epoch))

    for i, (X, l, Z, _) in enumerate(data_iterator):
        not_broken = l != -1
        X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
        if i % 20000 == 0:
            # Save clip for debugging
            clip = X[0].transpose(0, 1).reshape(3, -1, 112, 112)
            label = class_names[int(l[0])].replace('/', '_')
            batch2gif(clip, int(l[0]), opt.savename + '/samples/samples_train_epoch%d_iter%d_%s' % (epoch, i, label))
        batch_times.append(time.time() - tt_batch)
        s = list(X.shape)

        # Compute embeddings for input batch.
        tt_model = time.time()
        Y = model(X.to(opt.device))
        Y = Y[:s[0]]
        Z = Z.to(opt.device)

        # Compute Accuracy.
        pred_embed = Y.detach().cpu().numpy()
        pred_label = cdist(pred_embed, class_embedding, 'cosine').argmin(1)
        acc = accuracy_score(l.numpy(), pred_label) * 100
        accuracy_regressor.append(acc)

        # Compute loss.
        loss = criterion(Y, Z)
        optimizer.zero_grad()
        loss.backward()

        #Update weights using comp. gradients.
        optimizer.step()

        model_times.append(time.time() - tt_model)
        #Store loss per iteration.
        losses.append(loss.item())
        if i == len(train_dataloader)-1 or i*opt.bs > 100000:
            txwriter.add_scalar('Train/Loss', np.mean(losses), epoch)
            txwriter.add_scalar('Train/RegressorAccuracy', np.mean(accuracy_regressor), epoch)
            break

        tt_batch = time.time()

    print(Fore.RED, 'Train Accuracy: regressor {0:2.1f}%'.format(np.mean(accuracy_regressor)), Style.RESET_ALL)
    batch_times, model_times = np.sum(batch_times), np.sum(model_times)
    print('TOTAL time for: load the batch %.2f sec, run the model %.2f sec, train %.2f min' % (
                                    batch_times, model_times, (batch_times+model_times)/60))


"""========================================================="""


def evaluate(test_dataloader, txwriter, epoch):
    """
    This function is called every epoch to evaluate the model on 50% of the classes.
    """
    name = test_dataloader.dataset.name
    _ = model.eval()
    with torch.no_grad():
        ### For all test images, extract features
        n_samples = len(test_dataloader.dataset)

        predicted_embed = np.zeros([n_samples, 300], 'float32')
        true_embed = np.zeros([n_samples, 300], 'float32')
        true_label = np.zeros(n_samples, 'int')
        good_samples = np.zeros(n_samples, 'int') == 1

        final_iter = test_dataloader
        if 'features' not in opt.dataset and opt.progressbar:
            final_iter = tqdm(test_dataloader, desc='Extracting features...')

        fi = 0
        for idx, data in enumerate(final_iter):
            X, l, Z, _ = data
            not_broken = l != -1
            X, l, Z = X[not_broken], l[not_broken], Z[not_broken]
            if len(X) == 0: continue
            # Run network on batch
            Y = model(X.to(opt.device))
            Y = Y.cpu().detach().numpy()
            l = l.cpu().detach().numpy()
            predicted_embed[fi:fi + len(l)] = Y
            true_embed[fi:fi + len(l)] = Z.squeeze()
            true_label[fi:fi + len(l)] = l.squeeze()
            good_samples[fi:fi + len(l)] = True
            fi += len(l)

    predicted_embed = predicted_embed[:fi]
    true_embed, true_label = true_embed[:fi], true_label[:fi]

    # Calculate accuracy over test classes
    class_embedding = test_dataloader.dataset.class_embed
    accuracy, accuracy_top5 = compute_accuracy(predicted_embed, class_embedding, true_embed)

    # Logging using tensorboard
    txwriter.add_scalar(name+'/Accuracy', accuracy, epoch)
    txwriter.add_scalar(name+'/Accuracy_Top5', accuracy_top5, epoch)

    # Printing on terminal
    res_str = '%s Epoch %d: Test accuracy: %2.1f%%.' % (name.upper(), epoch, accuracy)
    # res_str = '\n%s Epoch %d: Test accuracy: %2.1f%%, Top5 %2.1f%%.' % (name.upper(), epoch, accuracy, accuracy_top5)

    # Logging accuracy in CSV file
    with open(opt.savename+'/'+name+'_accuracy.csv', 'a') as f:
        f.write('%d, %.1f,%.1f\n' % (epoch, accuracy, accuracy_top5))

    if opt.split == -1:
        # Calculate accuracy per split
        # Only when the model has been trained on a different dataset
        accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
        for split in range(len(accuracy_split)):
            # Select test set
            np.random.seed(split) # fix seed for future comparability
            sel_classes = np.random.permutation(len(class_embedding))[:len(class_embedding) // 2]
            sel = [l in sel_classes for l in true_label]
            test_classes = len(sel_classes)

            # Compute accuracy
            subclasses = np.unique(true_label[sel])
            tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
            acc, acc5 = compute_accuracy(predicted_embed[sel], class_embedding[sel_classes], true_embed[sel])
            accuracy_split[split] = acc
            accuracy_split_top5[split] = acc5

        # Printing on terminal
        res_str += ' -- Split accuracy %2.1f%% (+-%.1f) on %d classes' % (
                        accuracy_split.mean(), accuracy_split.std(), test_classes)
        accuracy_split, accuracy_split_std = np.mean(accuracy_split), np.std(accuracy_split)
        accuracy_split_top5, accuracy_split_top5_std = np.mean(accuracy_split_top5), np.std(accuracy_split_top5)

        # Logging using tensorboard
        txwriter.add_scalar(name+'/AccSplit_Mean', accuracy_split, epoch)
        txwriter.add_scalar(name+'/AccSplit_Std', accuracy_split_std, epoch)
        txwriter.add_scalar(name+'/AccSplit_Mean_Top5', accuracy_split_top5, epoch)
        txwriter.add_scalar(name+'/AccSplit_Std_Top5', accuracy_split_top5_std, epoch)

        # Logging accuracy in CSV file
        with open(opt.savename + '/' + name + '_accuracy_splits.csv', 'a') as f:
            f.write('%d, %.1f,%.1f,%.1f,%.1f\n' % (epoch, accuracy_split, accuracy_split_std,
                                                   accuracy_split_top5, accuracy_split_top5_std))
    print(Fore.GREEN, res_str, Style.RESET_ALL)
    return accuracy, accuracy_top5


def compute_accuracy(predicted_embed, class_embed, true_embed):
    """
    Compute accuracy based on the closest Word2Vec class
    """
    assert len(predicted_embed) == len(true_embed), "True and predicted labels must have the same number of samples"
    y_pred = cdist(predicted_embed, class_embed, 'cosine').argsort(1)
    y = cdist(true_embed, class_embed, 'cosine').argmin(1)
    accuracy = accuracy_score(y, y_pred[:, 0]) * 100
    accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, :5])]) * 100
    return accuracy, accuracy_top5


"""===================SCRIPT MAIN========================="""

if __name__ == '__main__':
    trainsamples = 0
    if not opt.evaluate:
        trainsamples = len(dataloaders['training'][0].dataset)
        with open(opt.savename + '/train_samples_%d_%d.txt' % (opt.n_classes, trainsamples), 'w') as f:
            f.write('%d, %d\n' % (opt.n_classes, trainsamples) )

    best_acc = 0
    print('\n----------')
    txwriter = SummaryWriter(logdir=opt.savename)
    epoch_times = []
    for epoch in range(opt.n_epochs):
        print('\n{} classes {} from {}, LR {} BS {} CLIP_LEN {} N_CLIPS {} OVERLAP {} SAMPLES {}'.format(
                    opt.network.upper(), opt.n_classes,
                    opt.dataset.upper(), opt.lr, opt.bs, opt.clip_len, opt.n_clips,
                    opt.class_overlap, trainsamples))
        print(opt.savename)
        tt = time.time()

        ## Train one epoch
        if not opt.evaluate:
            _ = model.train()
            train_one_epoch(dataloaders['training'][0], model, optimizer, criterion, opt, epoch)

        ### Evaluation
        accuracies = []
        for test_dataloader in dataloaders['testing']:
            accuracy, _ = evaluate(test_dataloader, txwriter, epoch)
            accuracies.append(accuracy)
        accuracy = np.mean(accuracies)

        if accuracy > best_acc:
            # Save best model
            torch.save({'state_dict': model.state_dict(), 'opt': opt, 'accuracy': accuracy},
                       opt.savename + '/checkpoint.pth.tar')
            best_acc = accuracy

        #Update the Metric Plot and save it.
        epoch_times.append(time.time() - tt)
        print('----- Epoch ', Fore.RED, '%d' % epoch, Style.RESET_ALL,
              'done in %.2f minutes. Remaining %.2f minutes.' % (
              epoch_times[-1]/60, ((opt.n_epochs-epoch-1)*np.mean(epoch_times))/60),
              Fore.BLUE, 'Best accuracy %.1f' % best_acc, Style.RESET_ALL)
        # scheduler.step(accuracy)
        scheduler.step()
        opt.lr = optimizer.param_groups[0]['lr']

        if opt.evaluate:
            break

    txwriter.close()

