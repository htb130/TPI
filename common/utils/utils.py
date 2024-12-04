import random
import torch
import os, pickle
import numpy as np
from tqdm import tqdm
import requests
import itertools
import multiprocessing
from torch import Tensor

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

def one_hot_to_index(one_hot: Tensor) -> Tensor:
    """
    Converts a one-hot tensor into a tensor with corresponding indexes
    """
    device, dtype = one_hot.device, one_hot.dtype
    vocab_size = one_hot.shape[-1]
    oh2idx = torch.tensor(range(vocab_size), dtype=dtype, device=device)
    return (one_hot @ oh2idx.unsqueeze(dim=1)).long().squeeze(dim=-1)

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)




def evaluate_loss(model, dataloader, loss_fn, text_field, epoch, device = 'cuda', args=None):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (images, captions) in enumerate(dataloader):
                # if it == 10:
                #     break
                captions = captions.to(device)
                if isinstance(images, tuple) or isinstance(images, list):
                    images = [x.to(device) for x in images]
                else:
                    images = images.to(device)
                out = model(images, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

def dict_to_cuda(input_dict, deivce):
    for key in input_dict:
        if isinstance(input_dict[key], list):
            input_dict[key] = [ val.to(deivce) for val in input_dict[key]]
        elif isinstance(input_dict[key], dict):
            dict_to_cuda(input_dict[key], deivce)
        else:
            input_dict[key] = input_dict[key].to(deivce)



def train_xe(model, dataloader, optim, loss_fn, text_field, epoch, device = 'cuda', scheduler = None, args=None):
    # Training with cross-entropy
    model.train()
    if scheduler is not None:
        scheduler.step()
    # print('lr0 = ', optim.state_dict()['param_groups'][0]['lr'])
    # print('lr1 = ', optim.state_dict()['param_groups'][1]['lr'])
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(dataloader)) as pbar:
        for it, (images, captions) in enumerate(dataloader):
            # if it == 10:
            #     break
            captions = captions.to(device)
            if isinstance(images, tuple) or isinstance(images, list):
                images = [x.to(device) for x in images]
            else:
                images = images.to(device)
            out = model(images, captions)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, out.shape[-1]), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            # if scheduler is not None:
            #     scheduler.step()

    loss = running_loss / len(dataloader)
    return loss
