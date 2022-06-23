
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += n * val
        self.avg = self.sum / self.count
def accuracy(output,label):
    _,pred=output.topk(1)
    pred=pred.t()
    # print(pred)
    # print(label)
    correct=pred.eq(label.unsqueeze(0).view_as(pred))
    acc=correct.data.sum()/correct.size()[1]
    return acc
