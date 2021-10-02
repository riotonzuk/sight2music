'''
Author: Rinav Kasthuri
Contains main module for Sight2Music model as well as various experiment modules.

Credit to Damon Gwinn for the base music transformer code:
    https://github.com/gwinndr/MusicTransformer-Pytorch/
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device
from .positional_encoding import PositionalEncoding
from .transformer import Transformer
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR


class Sight2Music(nn.Module):
    '''
    Sight2Music takes in an image and a music piece as inputs.
    It returns a similarity score between the two.
    '''
    
    def __init__(self, d_model=512):
        super().__init__()
        # can be any convolutional backbone, ResNet used for convenience
        self.img_interpreter = nn.Sequential(
                *(list(models.resnet50().children())[:-1]),
                nn.Flatten(),
                nn.LazyLinear(d_model)
        )
        
        self.music_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.music_interpreter = nn.LSTM(d_model, d_model, batch_first=True)
        self.music_reducer = nn.LazyLinear(REP_DIM)

    def forward(self, img, mid):
        # latent representation of image
        img_embedding = self.img_interpreter(img)
        music_embedding = self.music_embedding(mid)
        music_embedding = self.music_interpreter(music_embedding)[0]
        
        # latent representation of music
        music_embedding = self.music_reducer(music_embedding).mean(dim=1)
        
        # score range: [-1, 1]
        return F.cosine_similarity(img_embedding, music_embedding, dim=1)



class MusicTransformer(nn.Module):
    '''Author: Damon Gwinn'''
    
    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False):
        super().__init__()

        self.dummy      = DummyDecoder()
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr

        # midi embedding
        self.mid_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # base transformer
        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
        
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )

        self.lstm = nn.LSTM(512, 512)

        # Final output is a softmaxed linear layer
        self.Wout       = nn.LazyLinear(VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)


    def forward(self, mid, mask=True):
        batch_size = mid.shape[0]
        num_tokens = mid.shape[1]

        if mask:
            mask = self.transformer.generate_square_subsequent_mask(max(num_tokens, 1)).to(get_device())
        else:
            mask = None

        if num_tokens == 0:
            mid_embedded = self.mid_embedding(torch.randint(0, VOCAB_SIZE-1, (1, 1)).cuda())
            weights = None
        else:
            mid_embedded = self.mid_embedding(mid)

        # Change input shape to (max_seq, batch_size, d_model)
        mid_embedded = mid_embedded.permute(1,0,2)

        # represent positionality of each token
        pe_op = self.positional_encoding(mid_embedded)

        # since there are no true decoder layers, the tgt is unused
        # however, Pytorch wants src and tgt to have some equal dims
        transformer_op = self.transformer(src=pe_op, tgt=pe_op, src_mask=mask)
       
        # back to (batch_size, max_seq, d_model)
        transformer_op = transformer_op.permute(1,0,2)

        preds = self.Wout(transformer_op)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return preds


    # generate
    def generate(self, primer, target_seq_length=1024, beam=0, beam_chance=1.0):
        """
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        """

        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            
            y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :TOKEN_END]
            
            token_probs = y[:, cur_i-1, :]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

'''
Here's the thought process for the below Guide and Critic modules:
    Let's say the MusicTransformer composes a piece without any primer.
    The Guide can then try to guide the composed piece to fit better with
    the image and the Critic can then try to evaluate the final music piece
    with the image. This pipeline resembles that of a generative adversarial
    network. From my experiments, however, it doesn't seem to work well.
'''

class Guide(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mid_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.lstm = nn.LSTM(input_size=d_model+IMG_REP_DIM, hidden_size=d_model+IMG_REP_DIM,
                            proj_size=VOCAB_SIZE, batch_first=True)
        self.interpreter = nn.LazyLinear(VOCAB_SIZE)

    def forward(self, mid, img_rep):
        mid_embedded = self.mid_embedding(mid)
        img_rep = img_rep.expand(-1, len(mid_embedded[0]), -1)
        combined_rep = torch.cat([mid_embedded, img_rep], dim=-1)
        tailored_logits, _ = self.lstm(combined_rep)
        return tailored_logits


class Critic(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.rnn = nn.LSTM(input_size=d_model, hidden_size=d_model,
                           proj_size=IMG_REP_DIM if d_model > IMG_REP_DIM else 0, batch_first=True)
        self.music_evaluator = nn.LazyLinear(IMG_REP_DIM)
        self.embedding_translator = nn.LazyLinear(d_model)
        self.cos = nn.CosineSimilarity(dim=2)


    def forward(self, mid, img_rep, mid_embedded=None):
        if mid_embedded == None:
            mid_embedded = F.one_hot(mid, VOCAB_SIZE).float()
        else:
            mid_embedded = F.softmax(mid_embedded, dim=-1)

        mid_embedded = self.embedding_translator(mid_embedded)
        num_tokens = mid_embedded.shape[1]
        
        music_rep, (hidden, cell) = self.rnn(mid_embedded)
        weight_mask = 2 * torch.linspace(1, num_tokens, num_tokens)\
                               .unsqueeze(0).unsqueeze(-1).to(get_device()) / (num_tokens * (num_tokens + 1))
        ###combined_rep = self.music_evaluator(torch.cat([hidden.squeeze(0), cell.squeeze(0)], dim=-1))
        ###similarity_rating = self.cos(combined_rep, img_rep.squeeze(1))
        similarity_ratings = self.cos(music_rep, img_rep.expand(-1,num_tokens,-1)).unsqueeze(-1)
        similarity_rating = (similarity_ratings * weight_mask).sum(dim=1)
        return similarity_rating


'''
The Autoencoder is a fairly popular way to encode and decode sequences
for tasks like language translation, so the thought process here is to
encode the image and then decode it as a music sequence.

This doesn't seem to work well, however, for medium-length sequences (like
256 tokens long). An attempt was made to use a hierarchical decoding structure
based on https://arxiv.org/pdf/1803.05428.pdf, but that didn't seem to work well
either (perhaps an issue with this dataset specifically or with the code).
'''


class HierarchicalAutoencoder(nn.Module):
    def __init__(self, d_model=512, decod_seq=16, max_seq=256):
        super().__init__()

        ### HYPERPARAMETERS ###
        self.max_seq = max_seq
        self.decod_seq = decod_seq
        assert self.max_seq % self.decod_seq == 0
        self.d_model = d_model

        ### MAIN ###
        self.music_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.encoder = nn.LSTM(self.d_model, self.d_model, bidirectional=True, batch_first=True)
        self.enc_to_cond_embedding = nn.Sequential(
                nn.LazyLinear(2 * self.d_model),
                nn.LazyLinear(self.d_model),
                nn.Tanh()
        )
        self.conductor = nn.LSTM(self.d_model, self.d_model, num_layers=2, batch_first=True)
        self.cond_to_dec_embedding = nn.Sequential(
                nn.LazyLinear(2 * self.d_model),
                nn.LazyLinear(self.d_model),
                nn.Tanh()
        )
        self.decoder = nn.LSTM(VOCAB_SIZE + self.d_model, self.d_model, num_layers=2, batch_first=True)
        self.final_interpreter = nn.LazyLinear(VOCAB_SIZE)

    def forward(self, mid):
        batch_size = mid.shape[0]

        mid_embed = self.music_embedding(mid)
        _, hidden_enc = self.encoder(mid_embed)

        # convert encoder hidden state into latent representation
        latent = torch.cat(hidden_enc, dim=-1)
        latent = torch.cat([latent[0], latent[1]], dim=-1)
        latent = self.enc_to_cond_embedding(latent).unsqueeze(1)
        assert latent.shape == (batch_size, 1, self.d_model)
        
        # initialize conductor hidden state and output
        h0_cond, c0_cond = (torch.zeros(2, batch_size, self.d_model).to(get_device()),
                            torch.zeros(2, batch_size, self.d_model).to(get_device()))
        output = torch.zeros(batch_size, self.max_seq, VOCAB_SIZE).to(get_device())

        # for each embedding produced by conductor
        for cond_embedding_idx in range(self.max_seq // self.decod_seq):
            cond_embed, (h0_cond, c0_cond) = self.conductor(latent, (h0_cond, c0_cond))
            cond_embed = self.cond_to_dec_embedding(cond_embed)

            # initialize previous decoder token for each conductor embedding as zeros
            hidden_dec = h0_cond, c0_cond #(torch.zeros(2, batch_size, self.d_model).to(get_device()),
                         # torch.zeros(2, batch_size, self.d_model).to(get_device()))
            prev_dec_token = torch.zeros(batch_size, 1, VOCAB_SIZE).to(get_device())

            # for each prospective output token
            for i in range(self.decod_seq):
                prev_token = torch.cat([prev_dec_token, cond_embed], dim=-1)
                prev_dec_token, hidden_dec = self.decoder(prev_token, hidden_dec)
                prev_dec_token = self.final_interpreter(prev_dec_token)
                output[:, cond_embedding_idx * self.decod_seq + i] = prev_dec_token.squeeze(1)

        return output

    
class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model=512, max_seq=256):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq

        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
        self.encoder = Creator(d_model=self.d_model, output_dim=self.d_model, max_sequence=self.max_seq)
        self.decoder = Creator(d_model=self.d_model * 2, n_layers=2, num_heads=2, \
                               output_dim=self.d_model, max_sequence=self.max_seq)
        self.d_modeller = nn.LazyLinear(self.d_model)
        self.output = nn.LazyLinear(VOCAB_SIZE)

    def forward(self, mid):
        batch_size = mid.shape[0]
        mid_embedded = self.embedding(mid)
        _, (hidden, _) = self.lstm(mid_embedded)
        latent = self.lstm_reducer(torch.cat([hidden[0], hidden[1], hidden[2], hidden[3]], dim=-1))
        latent = latent.unsqueeze(1)
        assert latent.shape == (batch_size, 1, self.d_model)
        tokens = torch.zeros(batch_size, 1, self.d_model * 2).to(get_device())
        
        for i in range(self.max_seq):
            trans_input = torch.cat([self.d_modeller(tokens), latent.expand(-1,i+1,-1)], dim=-1)
            trans_output = self.transformer(trans_input, is_embedding=True)
            tokens = torch.cat([tokens, trans_output[:,-1].unsqueeze(1)], dim=1)

        output = self.output(tokens[:,1:])

        return output






# MusicTransformer
class MusicTransformer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False):
        super(MusicTransformer, self).__init__()

        self.dummy      = DummyDecoder()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence + 1
        self.rpr        = rpr

        # midi embedding
        self.mid_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        # Base transformer
        if(not self.rpr):
            # To make a decoder-only transformer we need to use masked encoder layers
            # Dummy decoder to essentially just return the encoder output
            self.transformer = Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
        
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )
            
        # Final output is a softmaxed linear layer
        self.Wout       = nn.LazyLinear(VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)


    # forward
    def forward(self, mid, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """
        
        batch_size = mid.shape[0]
        num_tokens = mid.shape[1]

        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(num_tokens + (1 if num_tokens == 0 else 0)).to(get_device())
        else:
            mask = None
         
        # if we want a completely new piece from the image,
        # start with no impact from music
        if num_tokens == 0:
            num_tokens = 1
            new_shape = [batch_size, num_tokens, self.d_model]
            mid_embedded = torch.zeros(new_shape).cuda()
        else:
            # first get music embedding
            mid_embedded = self.mid_embedding(mid)

        # Change input shape to (max_seq, batch_size, d_model)
        x = mid_embedded.permute(1,0,2)

        # REMEMBER TO UNCOMMENT LATER
        assert x.shape == (num_tokens, batch_size, self.d_model), f"{x.shape} != {(num_tokens, batch_size, self.d_model)}"

        pe_op = self.positional_encoding(x)

        # since there are no true decoder layers, the tgt is unused
        # however, Pytorch wants src and tgt to have some equal dims
        transformer_op = self.transformer(src=pe_op, tgt=pe_op, src_mask=mask)

        #x = self.Wout(torch.cat([transformer_op, emotion_embedded], dim=-1))
        x = self.Wout(transformer_op)

        # back to (batch_size, max_seq, VOCAB_SIZE)
        x = x.permute(1,0,2)
       
        del mask

        # REMEMBER TO UNCOMMENT LATER
        assert x.shape == (batch_size, num_tokens, VOCAB_SIZE), f"{x.shape} != {(batch_size, num_tokens, self.d_model)}"

        # They are trained to predict the next note in sequence (we don't need the last one)
        return x


    # generate
    def generate(self, img, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Generates midi given a primer sample. Music can be generated using a probability distribution over
        the softmax probabilities (recommended) or by using a beam search.
        ----------
        """

        assert (not self.training), "Cannot generate while in training mode"

        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())


        # print("primer:",primer)
        # print(gen_seq)
        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            
            y = self.softmax(self.forward(gen_seq[..., :cur_i], img))[..., :TOKEN_END]
            token_probs = y[:, cur_i-1, :]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                # print("next token:",next_token)
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

# Used as a dummy to nn.Transformer
# DummyDecoder
class DummyDecoder(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    A dummy decoder that returns its input. Used to make the Pytorch transformer into a decoder-only
    architecture (stacked encoders with dummy decoder fits the bill)
    ----------
    """

    def __init__(self):
        super(DummyDecoder, self).__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Returns the input (memory)
        ----------
        """

        return memory

