'''
Authors: Damon Gwinn (base code), Rinav Kasthuri (edits for Sight2Music)
'''

import torch
import torch.nn as nn
import torchvision.models as models
import os
import random
import pretty_midi
import processor

from processor import encode_midi, decode_midi

from utilities.argument_funcs import parse_generate_args, print_generate_args
from model.sight2music import * 
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy, process_midi, get_img
from torch.utils.data import DataLoader
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point. Generates music from a model specified by command line arguments
    ----------
    """

    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    # Grabbing dataset if needed
    dataset, *_ = create_epiano_datasets(args.midi_root, 2048, random_seq=False)
    dataloader = DataLoader(dataset, 4)

    creator = Creator(n_layers=args.n_layers, num_heads=args.num_heads,
                d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                max_sequence=args.max_sequence, rpr=args.rpr).to(get_device())
    creator.load_state_dict(torch.load(f"cloud_weights/query/creator_{args.model_weights}"))

    queryer = SimilarityNet(d_model=args.d_model).to(get_device())
    queryer.load_state_dict(torch.load(f"cloud_weights/query/queryer_{args.model_weights}"))
    
    # get conditioning image
    img = get_img(args.image_path, augment=False).unsqueeze(0).to(get_device())
    
    # GENERATION
    creator.eval()
    queryer.eval()
    query_results = []
    
    with torch.no_grad():
        for music_pieces, *_ in dataloader:
                music_pieces = music_pieces.to(get_device())
                query_results.extend(queryer(img.repeat(len(music_pieces), 1, 1, 1), music_pieces).tolist())
        match_idx = torch.tensor(query_results).argmax()
        primer, *_ = dataset[match_idx]
        print(f"MATCH INDEX: {match_idx}")

        # Saving primer first
        f_path = os.path.join(args.output_dir, "primer.mid")
        decode_midi(primer[:args.num_prime].cpu().numpy(), file_path=f_path)

        if(args.beam > 0):
            print("BEAM:", args.beam)
            beam_seq = creator.generate(primer[:args.num_prime], args.target_seq_length, beam=args.beam)

            f_path = os.path.join(args.output_dir, "beam.mid")
            decode_midi(beam_seq[0].cpu().numpy(), file_path=f_path)
        else:
            print("RAND DIST")
            orig_seq = creator.generate(primer[:args.num_prime], args.target_seq_length, beam=0)
            f_orig_path = os.path.join(args.output_dir, "rand.mid")
            decode_midi(orig_seq[0].cpu().numpy(), file_path=f_orig_path)




if __name__ == "__main__":
    main()
