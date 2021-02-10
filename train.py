#!/usr/bin/env python
"""Train a CNN for Google speech commands."""

#import argparse
import time
from opts import *
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision
from torchvision.transforms import *

from tensorboardX import SummaryWriter

from datasets import *
from transforms import *
from mixup import *

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']



def train(epoch,model,train_dataloader, criterion,optimizer,use_gpu,writer):
    global global_step

    print("epoch %3d with lr=%.02e" % (epoch, get_lr(optimizer)))
    phase = 'train'
    writer.add_scalar('%s/learning_rate' % phase,  get_lr(optimizer), epoch)

    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        if opt['mixup']:
            inputs, targets = mixup(inputs, targets, num_classes=len(CLASSES))

        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward/backward
        outputs = model(inputs)
        if opt['mixup']:
            loss = mixup_cross_entropy_loss(outputs, targets)
        else:
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.item() * inputs.size(0) #loss.data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        if opt['mixup']:
            targets = batch['target']
            targets = Variable(targets, requires_grad=False).cuda()
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss.item() * inputs.size(0), global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

def valid(epoch,model,valid_dataloader,criterion,optimizer,use_gpu, start_timestamp,writer,full_name):
    global best_accuracy, best_loss, global_step

    phase = 'valid'
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(valid_dataloader, unit="audios", unit_scale=valid_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        it += 1
        global_step += 1
        running_loss += loss.data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        writer.add_scalar('%s/loss' % phase, loss.data[0], global_step)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it
    writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
    writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

    checkpoint = {
        'epoch': epoch,
        'step': global_step,
        'state_dict': model.state_dict(),
        'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer' : optimizer.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/best-loss-speech-commands-checkpoint-%s.pth' % full_name)
        torch.save(model, '%d-%s-best-loss.pth' % (start_timestamp, full_name))
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(checkpoint, 'checkpoints/best-acc-speech-commands-checkpoint-%s.pth' % full_name)
        torch.save(model, '%d-%s-best-acc.pth' % (start_timestamp, full_name))

    torch.save(checkpoint, 'checkpoints/last-speech-commands-checkpoint.pth')
    del checkpoint  # reduce memory

    return epoch_loss

def main(opt):
    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)
    if use_gpu:
        torch.backends.cudnn.benchmark = True

    if opt['input']=='mel32':
        n_mels = 32
    if opt['input']== 'mel40':
        n_mels = 40

    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(opt['background_noise'], data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_dataset = SpeechCommandsDataset(opt['train_dataset'],
                                    Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))

    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    valid_dataset = SpeechCommandsDataset(opt['valid_dataset'],
                                    Compose([LoadAudio(),
                                             FixAudioLength(),
                                             valid_feature_transform]))

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_dataset, batch_size=opt['batch_size'], sampler=sampler,
                                  pin_memory=use_gpu, num_workers=opt['dataload_workers_nums'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=opt['batch_size'], shuffle=False,
                                  pin_memory=use_gpu, num_workers=opt['dataload_workers_nums'])


    # a name used to save checkpoints etc.
    full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (opt['model'], opt['optim'], opt['lr_scheduler'], opt['batch_size'], opt['learning_rate'], opt['weight_decay'])
    if opt['comment']:
        full_name = '%s_%s' % (full_name, opt['comment'])

    model = models.create_model(model_name=opt['model'], num_classes=len(CLASSES), in_channels=1)
    print("model name",opt['model'])
    print(model)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if opt['optim'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt['learning_rate'], momentum=0.9, weight_decay=opt['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'])

    start_timestamp = int(time.time()*1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    if opt['resume']:
        print("resuming a checkpoint '%s'" % opt['resume'])
        checkpoint = torch.load(opt['resume'])
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce memory

    if opt['lr_scheduler'] == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt['lr_scheduler_patience'], factor=opt['lr_scheduler_gamma'])
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt['lr_scheduler_step_size'], gamma=opt['lr_scheduler_gamma'], last_epoch=start_epoch-1)

    writer = SummaryWriter(comment=('_speech_commands_' + full_name))
    print("training %s for Google speech commands..." % opt['model'])
    since = time.time()
    for epoch in range(start_epoch, opt['max_epochs']):
        if opt['lr_scheduler'] == 'step':
            lr_scheduler.step()

        train(epoch,model,train_dataloader, criterion,optimizer,use_gpu,writer)
        epoch_loss = valid(epoch,model,valid_dataloader,criterion,optimizer,use_gpu, start_timestamp,writer,full_name)

        if opt['lr_scheduler'] == 'plateau':
            lr_scheduler.step(metrics=epoch_loss)
    
        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
        print("%s, best accuracy: %.02f%%, best loss %f" % (time_str, 100*best_accuracy, best_loss))
    print("finished")

if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)
    main(opt)
