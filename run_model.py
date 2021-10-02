import torch
import torch.nn.functional as F
import time

from processor import *
from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr

from torch.nn import CrossEntropyLoss
from dataset.e_piano import compute_epiano_accuracy


CE = CrossEntropyLoss(ignore_index=TOKEN_PAD)

# train_epoch
def train_epoch(cur_epoch, models, dataloader, losses, opts, lr_schedulers):
    creator, queryer = models
    opt_creator, opt_queryer = opts
    sched_creator, sched_queryer = lr_schedulers if lr_schedulers is not None else [None] * 2

    creator.train()
    queryer.train()

    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()


        #### DATA EXTRACTION ####
        original_piece = batch[0].to(get_device())
        match_img = batch[1].to(get_device())
        target_piece = batch[2].to(get_device())
        critic_piece = batch[3].to(get_device())
        critic_target_piece = batch[4].to(get_device())


        #### TRAINING ####
        '''
        opt_creator.zero_grad()
        created_piece = creator(original_piece)
        ce_loss = CE(created_piece.flatten(0,1), target_piece.flatten())
        ce_loss.backward()
        opt_creator.step()
        '''
        ce_loss = torch.zeros(1)

        opt_queryer.zero_grad()
        output = queryer(match_img, critic_piece)
        query_loss = (1 - output).mean()
        query_loss.backward()
        opt_queryer.step()
        

        #### UPDATE LRs ####
        if lr_schedulers is not None:
            sched_creator.step()
            sched_queryer.step()


        #### STAT DISPLAY ####
        time_after = time.time()
        time_took = time_after - time_before
        
        print(SEPERATOR)
        print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
        print(f"LR: creator = {get_lr(opt_creator)}, queryer = {get_lr(opt_queryer)}")  
        print(f"Train loss: CE = {ce_loss.item()}, cos = {query_loss.item()}") 
        print("")
        print("Time (s):", time_took)
        print(SEPERATOR)
        print("")


# eval_model
def eval_model(model, dataloader):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Evaluates the model and prints the average loss and accuracy
    ----------
    """
    
    model.eval()

    sum_loss = 0.

    with torch.no_grad():
        n_test      = len(dataloader)

        for batch in dataloader:
            original_piece = batch[-6].to(get_device())
            target_piece = batch[-3].to(get_device())
            created_piece = model(original_piece)
            sum_loss += CE(created_piece.flatten(0,1), target_piece.flatten()).item()

    avg_loss = sum_loss / n_test
    
    return avg_loss
