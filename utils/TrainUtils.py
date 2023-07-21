import os
import logging
import torch

def create_directory(new_dir):
    try:
        os.mkdir(new_dir)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

def extract_batch(test_loader_ensemble_iterator, test_loader_ensemble):
    try:
        Unsup_batch = next(test_loader_ensemble_iterator)
    except StopIteration:
        test_loader_ensemble_iterator = iter(test_loader_ensemble)
        Unsup_batch = next(test_loader_ensemble_iterator)
    return Unsup_batch, test_loader_ensemble_iterator    
   

def UnNormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    tensor1 = torch.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        subtensor = tensor[i,:,:,:]
        for t, m, s in zip(subtensor, mean, std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        tensor1[i, :, :, :] = subtensor    
    return tensor1


class polynomial_LR():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, base_lr, max_iter, power=0.9):

        self._optimizer = optimizer
        self.base_lr = base_lr
        self.max_iter = max_iter
        self.power = power
        self.n_steps = 0


    # def step_and_update_lr(self):
    #     "Step with the inner optimizer"
    #     self._update_learning_rate()
    #     self._optimizer.step()


    # def zero_grad(self):
    #     "Zero out the gradients with the inner optimizer"
    #     self._optimizer.zero_grad()

    def update_lr(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.base_lr * ((1 - float(self.n_steps) / self.max_iter) ** self.power)    

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr  

              