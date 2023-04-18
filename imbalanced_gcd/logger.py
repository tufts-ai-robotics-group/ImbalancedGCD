from torch.utils.tensorboard import SummaryWriter


class AverageMeter():
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


class AverageWriter():
    def __init__(self, *args, **kwargs) -> None:
        self.writer = SummaryWriter(*args, **kwargs)
        self.scalars = {}

    def get_avg(self, tag):
        return self.scalars[tag].avg

    def update(self, tag, scalar_value, cnt=1):
        """Update scalar. Use cnt=1 and only update once per epoch for non-mean values.
        Args:
            tag (str): Scalar tag.
            scalar_value (float): Scalar value batch mean.
            cnt (int, optional): Number of samples for batch mean. Defaults to 1.
        """
        if tag not in self.scalars:
            self.scalars[tag] = AverageMeter()
        self.scalars[tag].update(scalar_value, cnt)

    def write(self, global_step):
        """Write scalars from epoch to Tensorboard and clear averages
        Args:
            global_step (int): Epoch number to record values for.
        """
        for tag, av_meter in self.scalars.items():
            self.writer.add_scalar(tag, av_meter.avg, global_step)
        self.scalars = {}
