import torch
import torch.nn.functional as F
import os
import time
import torch.nn as nn
import torchvision.models as models
import random

from dataset.e_piano import create_epiano_datasets, get_img, get_sampling_weights
from torch.utils.data import Dataset, DataLoader, \
                             WeightedRandomSampler, BatchSampler
from utilities.device import get_device
from utilities.constants import *

   
def _evaluate(model, dataset, loss_fn, eval_type="test"):
    model.eval()
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0

        for idx, data in enumerate(loader):
            _, image, _, _, _, truth_categs = data
            image = image.to(get_device())
            truth_categs = truth_categs.to(get_device())

            pred_categs = model(image)

            loss = loss_fn(pred_categs.flatten(), truth_categs.float())
            total_correct += ((pred_categs >= 0.).int().flatten() ==  truth_categs).sum()
            total_loss += loss.item()

        print(f"{eval_type} loss: {total_loss / (idx + 1)}")
        print(f"{eval_type} accuracy: {total_correct / len(dataset)}")
    
    
### PRINT FOR DEBUGGING ###
class PrintLayer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


def main():
    INPUT_DIR = "dataset/e_piano"

    
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(INPUT_DIR, 1)
    train_size = len(train_dataset)
    test_size = len(test_dataset)


    print(f"TRAIN SIZE: {train_size}, TEST SIZE: {test_size}")
    # create heavier weighting for happy / angry pictures
    train_sampling_weights = get_sampling_weights(txt_files=[v[1] for v in train_dataset.triplets])

    train_weighted_sampler = \
                WeightedRandomSampler(train_sampling_weights, train_size)
    train_batch_sampler = \
                BatchSampler(train_weighted_sampler, batch_size=4, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    

    '''
            nn.Sequential(
                nn.LazyConv2d(16, kernel_size=(3, 3)),
                nn.LazyConv2d(16, kernel_size=(3, 3)),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.LazyConv2d(32, kernel_size=(3, 3)),
                nn.LazyConv2d(32, kernel_size=(3, 3)),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.LazyConv2d(64, kernel_size=(3, 3)),
                nn.LazyConv2d(128, kernel_size=(3, 3)),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, 2),
                nn.ReLU(), 
                nn.LazyConv2d(256, kernel_size=(3, 3)),
                nn.LazyConv2d(512, kernel_size=(3, 3)),
                nn.BatchNorm2d(512),
                nn.ReLU(), 
                nn.Flatten(),
                nn.LazyLinear(REP_DIM),
            ),
    '''


    conv_backbone = nn.Sequential( 
            nn.Sequential(
                *(list(models.resnet50().children())[:-1]),
                nn.Flatten(),
                nn.LazyLinear(REP_DIM)
            ), 
            nn.LazyLinear(1)
    ).to(get_device())

    opt = torch.optim.Adam(conv_backbone.parameters(), lr=1e-5)
    embedding_loss_fn = nn.CosineEmbeddingLoss(margin=-1.)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    NUM_EPOCHS = 100

    print()
    print(f"using: {get_device()}")
    print()

    for epoch in range(NUM_EPOCHS):
        #### TRAINING ####
        conv_backbone.train()

        for idx, data in enumerate(train_loader):
            print(f"EPOCH {epoch}")
            start_time = time.time()
            opt.zero_grad()
            before_data_time = time.time()
            _, image, _, _, _, truth_categs = data
            image = image.to(get_device())
            truth_categs = truth_categs.to(get_device())
            after_data_time = time.time()
            print(f"TIME FOR DATA PROCESSING: {after_data_time - before_data_time}s")
            before_training_time = time.time()
            pred_embeddings = conv_backbone[0](image)
            pred_categs = conv_backbone[1](pred_embeddings)
            offset_embeddings = pred_embeddings.roll(1,0)
            offset_truth_categs = truth_categs.roll(1,0)
            embed_loss = embedding_loss_fn(pred_embeddings, offset_embeddings, \
                                    (torch.where((truth_categs==offset_truth_categs), 1, -1)))
            print(embed_loss)
            loss = embed_loss + bce_loss_fn(pred_categs.flatten(), truth_categs.float())
 
            loss.backward()
            opt.step()
            after_training_time = time.time()
            print(f"TIME FOR TRAINING: {after_training_time - before_training_time}s")

            print(f"loss for batch {idx} / {len(train_loader)}: {loss.item()}")
            
            end_time = time.time()
            print(f"OVERALL TIME: {end_time - start_time}s")
            print()


        #### TESTING ####
        if epoch % 10 == 0: 
            print("-------------------------------")
            print("-------------------------------")
            print(f"EPOCH: {epoch}")
            
            _evaluate(conv_backbone, train_dataset, bce_loss_fn, eval_type="train")
            _evaluate(conv_backbone, test_dataset, bce_loss_fn, eval_type="test")
            
            print("--------------------------------")
            print("-------------------------------")
            print()
            
            torch.save(conv_backbone.state_dict(), os.path.join("conv_weights", f"epoch_{epoch}.pickle"))
        
if __name__ == '__main__':
    main()
