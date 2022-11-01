import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import time
import math

from config import *
from data import Data
from transformer import Transformer
from utils import *


data = Data()
data.load_data(batch_size)

model = Transformer(
    src_vocab_size=data.src_vocab_size,
    tgt_vocab_size=data.tgt_vocab_size,
    src_pad_idx=data.src_pad_idx,
    tgt_pad_idx=data.tgt_pad_idx,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    d_model=d_model,
    n_head=n_head,
    n_hidden=n_hidden,
    drop_rate=drop_rate
)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'model trainable param count is: {param_count}')

optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=init_lr,
    betas=(0.9, 0.98),
    eps=1e-11,
)
lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(step, d_model, factor, warmup)
)

criterion = nn.CrossEntropyLoss(ignore_index=data.tgt_pad_idx) #TODO: label smoothing and kl

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.trg[:, :-1]
        tgt_y = batch.trg[:, 1:]

        out = model(src, tgt) # [batch_size, seq_len, tgt_vocab_size]
        out = out.reshape(-1, out.shape[-1])

        loss = criterion(out, tgt_y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()

        #lr_scheduler.step()

        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt[:, :-1]
            tgt_y = batch.tgt[:, 1:]

            out = model(src, tgt)
            out = out.reshape(-1, out.shape[-1])

            loss = criterion(out, tgt_y)
            epoch_loss += loss.item()

    #         total_bleu = []
    #         for j in range(batch_size):
    #             try:
    #                 trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
    #                 output_words = output[j].max(dim=1)[1]
    #                 output_words = idx_to_word(output_words, loader.target.vocab)
    #                 bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
    #                 total_bleu.append(bleu)
    #             except:
    #                 pass

    #         total_bleu = sum(total_bleu) / len(total_bleu)
    #         batch_bleu.append(total_bleu)

    # batch_bleu = sum(batch_bleu) / len(batch_bleu)
    # return epoch_loss / len(iterator), batch_bleu

    return epoch_loss / len(iterator)

def run(n_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for epoch in range(n_epoch):
        start_time = time.time()
        train_loss = train(model, data.train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, data.valid_iter, criterion)
        end_time = time.time()

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
    #bleus.append(bleu)
    #epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
          best_loss = valid_loss
          torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

    #print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
    #print(f'\tBLEU Score: {bleu:.3f}')

if __name__ == '__main__':
    run(n_epoch=num_epochs, best_loss=inf)
  