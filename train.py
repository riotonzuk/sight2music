import os
import csv
import shutil
import torch
import torch.nn as nn
import pickle
from processor import START_IDX
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, BatchSampler, WeightedRandomSampler
from torch.optim import Adam
import torchvision.models as models

from dataset.e_piano import create_epiano_datasets, get_sampling_weights, compute_epiano_accuracy

from model.sight2music import *
from model.loss import SmoothCrossEntropyLoss

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params
from utilities.run_model import train_epoch, eval_model

CSV_HEADER = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Eval loss", "Eval accuracy"]

# Baseline is an untrained epoch that we evaluate as a baseline loss and accuracy
BASELINE_EPOCH = -1

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Trains a model specified by command line arguments
    ----------
    """
    
    args = parse_train_args()
    print_train_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, "results")
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")
    best_acc_file = os.path.join(results_folder, "best_acc_weights.pickle")
    best_text = os.path.join(results_folder, "best_epochs.txt")

    ##### Tensorboard #####
    if(args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter

        tensorboad_dir = os.path.join(args.output_dir, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.input_dir, args.max_sequence, random_seq=True)

    train_sampling_weights = get_sampling_weights(os.path.join(args.input_dir, 'train'))
    train_sampler = WeightedRandomSampler(train_sampling_weights, len(train_sampling_weights))
    batch_sampler = BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=True)

    train_loader = DataLoader(train_dataset, num_workers=args.n_workers, batch_sampler=batch_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if(args.ce_smoothing is None):
        train_loss_func = eval_loss_func
    else:
        train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    creator = Creator(n_layers=args.n_layers, num_heads=args.num_heads,
                      d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                      max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())
    queryer = SimilarityNet(d_model=args.d_model).to(get_device())
     
    if args.continue_weights is None:
        loaded_emotion_detector_dict = torch.load("conv_weights/epoch_90.pickle")
        loaded_encoder_dict = {".".join(k.split(".")[1:]):v for k,v in loaded_emotion_detector_dict.items() \
                               if int(k.split(".")[0]) != 1}   
        queryer.img_interpreter.load_state_dict(loaded_encoder_dict)
    else:
        queryer.img_interpreter.load_state_dict(torch.load(f"rpr/weights/conv_{args.continue_weights}"))

    # TODO: remove this line for more flexibility
    queryer.img_interpreter.requires_grad_(False)

    loaded_models = [creator, queryer]
    

    print("cuda is available:", torch.cuda.is_available())
    ##### Continuing from previous training session #####
    start_epoch = BASELINE_EPOCH
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            return
        else: 
            creator.load_state_dict(torch.load(f"rpr/weights/creator_{args.continue_weights}"))
            queryer.load_state_dict(torch.load(f"rpr/weights/queryer_{args.continue_weights}"))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        return

    ##### Lr Scheduler vs static lr #####
    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)

        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr



    ##### Optimizer #####
    opts = [Adam(model.parameters(), lr=lr) for model in loaded_models]

    if(args.lr is None):
        lr_schedulers = [LambdaLR(opt, lr_stepper.step) for opt in opts]
    else:
        lr_schedulers = None

    ##### Tracking best evaluation accuracy #####
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)


    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
        if(epoch > BASELINE_EPOCH):
            print(SEPERATOR)
            print("NEW EPOCH:", epoch+1)
            print(SEPERATOR)
            print("")

            # Train
            train_epoch(epoch+1, loaded_models, train_loader, train_loss_func, opts, lr_schedulers)

            print(SEPERATOR)
            print("Evaluating:")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")
        
        if epoch % 10 == 0:
            # Eval
            train_loss = 0 # TODO: eval_model(creator, train_loader)
            eval_loss = 0 # TODO: eval_model(creator, test_loader)

            print("Epoch:", epoch+1)
            print("Avg train loss:", train_loss)
            print("Avg eval loss:", eval_loss)
            print(SEPERATOR)
            print("")
        
        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            creator_path = os.path.join(weights_folder, "creator_epoch_" + epoch_str + ".pickle")  
            queryer_path = os.path.join(weights_folder, "queryer_epoch_" + epoch_str + ".pickle")
            torch.save(creator.state_dict(), creator_path)
            torch.save(queryer.state_dict(), queryer_path)
    

if __name__ == "__main__":
    main()
