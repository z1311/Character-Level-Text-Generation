import string
import numpy as np
import torch
import torch.nn as nn

chars = string.ascii_letters + string.digits + string.punctuation + ' ' + '\n'
char2int = {ch:i for i,ch in enumerate(chars)}
int2char = {i:ch for i,ch in enumerate(chars)}
char_len = len(chars)

def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
def decodeBasedOnTopK(one_hot_pred, top_k=1):
    choices = np.argpartition(one_hot_pred, -top_k)[-top_k:]
    probs = one_hot_pred[choices]
    probs = softmax(probs)
    probs = probs / np.sum(probs)
    choice = np.random.choice(choices, 1, p=probs)[0]
    return choice

def sample(model, initial_letters = "Dread it", num_letters = 200, topk=5, device='cpu'):
    model.to(device)
    model.eval()
    state = model.init_hidden(1)

    #getting hidden states for initial letters
    for char in initial_letters:
        one_hot = torch.zeros((1, 1, char_len), dtype=torch.float, device=device)
        one_hot[0,0,char2int[char]] = 1
        _, state = model(one_hot, state)
    
    predict_letters = initial_letters[-1]

    for i in range(num_letters):
        one_hot = torch.zeros((1, 1, char_len), dtype=torch.float, device=device)
        one_hot[0,0,char2int[predict_letters[-1]]] = 1
        out, state = model(one_hot, state)
        predict_letters += int2char[decodeBasedOnTopK(out[0].cpu().data.numpy(), top_k=topk)]

    print("{} -- {}".format(initial_letters, predict_letters[1:]))