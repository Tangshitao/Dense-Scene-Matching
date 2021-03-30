import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, cfg, model=None, input_list=None):
        self.tb_writer = SummaryWriter(cfg.path)
        if model is not None:
            self.tb_writer.add_graph(model, input_list)

    def add_metrics(self, stage, niter, metrics):
        for k, v in metrics.items():
            self.tb_writer.add_scalar("{}/{}".format(stage, k), v, niter)


class Counter(object):
    def __init__(self):
        self.counter = {}

    def add_value(self, name, value, window_len=30):
        if name not in self.counter:
            self.counter[name] = AverageMeter(name, window_len)

        if type(value) is torch.Tensor:
            value = value.detach().cpu().numpy()
        self.counter[name].update(value)

    def items(self):
        return [(k, v.get_avg_value()) for k, v in self.counter.items()]

    def __setitem__(self, name, value):
        if name not in self.counter:
            self.counter[name] = AverageMeter(name)

        if type(value) is torch.Tensor:
            value = value.detach().cpu().numpy()

        self.counter[name].update(value)

    def __getitem__(self, name):
        return self.get_avg_value_by_name(name)

    def __contains__(self, name):
        return name in self.counter

    def __repr__(self):
        _str = ""
        for k, v in self.counter.items():
            _str = _str + str(v) + " ({})\n".format(self.get_median_value_by_name(k))
        return _str

    def get_smooth_value_by_name(self, name):
        return self.counter[name].get_smooth_value()

    def get_avg_value_by_name(self, name):
        return self.counter[name].get_avg_value()

    def get_median_value_by_name(self, name):
        return self.counter[name].get_median_value()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, window_len, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.window_len = window_len
        self.values = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val is None:
            return
        self.val = val

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.values.append(val)
        if self.window_len > 0 and len(self.values) > self.window_len:
            self.values = self.values[-self.window_len :]

    def get_smooth_value(self):
        return sum(self.values) / len(self.values)

    def get_avg_value(self):
        return self.avg

    def get_median_value(self):
        return np.median(self.values)

    def __str__(self):
        fmtstr = "{name} {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)
