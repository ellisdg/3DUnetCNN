"""
Modified from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
import shutil
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

try:
    from torch.utils.data._utils.collate import default_collate
except ModuleNotFoundError:
    # import from older versions of pytorch
    from torch.utils.data.dataloader import default_collate


def epoch_training(train_loader, model, criterion, optimizer, epoch, n_gpus=None, print_frequency=1,
                   print_gpu_memory=False, scaler=None, samples_per_epoch=None, iteration=1):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    if iteration > 1:
        prefix = "Epoch: [{}({})]".format(epoch, iteration)
    else:
        prefix = "Epoch: [{}]".format(epoch)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix=prefix)

    use_amp = scaler is not None

    # switch to train mode
    model.train()

    end = time.time()
    for i, item in enumerate(train_loader):
        images = item["image"]
        target = item["label"]
        # measure data loading time
        data_time.update(time.time() - end)

        if n_gpus:
            torch.cuda.empty_cache()
            if print_gpu_memory:
                for i_gpu in range(n_gpus):
                    print("Memory allocated (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.memory_allocated(i_gpu)))
                    print("Max memory allocated (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.max_memory_allocated(i_gpu)))
                    print("Memory cached (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.memory_cached(i_gpu)))
                    print("Max memory cached (device {}):".format(i_gpu),
                          human_readable_size(torch.cuda.max_memory_cached(i_gpu)))

        optimizer.zero_grad()
        loss, batch_size = batch_loss(model, images, target, criterion, n_gpus=n_gpus, use_amp=use_amp)

        # measure accuracy and record loss
        losses.update(loss.item(), batch_size)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # compute gradient and do step
            loss.backward()
            optimizer.step()

        del loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_frequency == 0:
            progress.display(i+1)

        if samples_per_epoch and (i + 1) * batch_size >= samples_per_epoch:
            break
    return losses.avg


def batch_loss(model, images, target, criterion, n_gpus=0, use_amp=None, inferer=None):
    if n_gpus is not None:
        images = images.cuda()
        target = target.cuda()
    # compute output
    if use_amp:
        from torch.cuda.amp import autocast
        with autocast():
            return _batch_loss(model, images, target, criterion, inferer=inferer)
    else:
        return _batch_loss(model, images, target, criterion, inferer=inferer)


def _batch_loss(model, images, target, criterion, inferer=None):
    """
    inferer: should take in the inputs and the model and output the prediction. This is based on the MONAI Inferer
    classes.
    """
    if inferer is not None:
        output = inferer(images, model).to(images.device)
    else:
        output = model(images)
    batch_size = images.size(0)
    loss = criterion(output, target)
    return loss, batch_size


def epoch_validation(val_loader, model, criterion, n_gpus, print_freq=1, use_amp=False, inferer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Validation: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, item in enumerate(val_loader):
            images = item["image"]
            target = item["label"]
            loss, batch_size = batch_loss(model, images, target, criterion, n_gpus=n_gpus,  use_amp=use_amp,
                                          inferer=inferer)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i+1)

            if n_gpus:
                torch.cuda.empty_cache()

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def human_readable_size(size, decimal_places=1):
    for unit in ['', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def collate_flatten(batch, x_dim_flatten=5, y_dim_flatten=2):
    x, y = default_collate(batch)
    if len(x.shape) > x_dim_flatten:
        x = x.flatten(start_dim=0, end_dim=len(x.shape) - x_dim_flatten)
    if len(y.shape) > y_dim_flatten:
        y = y.flatten(start_dim=0, end_dim=len(y.shape) - y_dim_flatten)
    return [x, y]


def collate_5d_flatten(batch, dim_flatten=5):
    return collate_flatten(batch, x_dim_flatten=dim_flatten, y_dim_flatten=dim_flatten)
