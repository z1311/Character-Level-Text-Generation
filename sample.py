import torch
import numpy as np
import argparse, string

from models import RNNModel, LSTMModel, GRUModel

def init():
    '''
    Initialization
    '''
    global device, char_len, char2int, int2char

    chars = string.ascii_letters + string.digits + string.punctuation + ' ' + '\n'
    char_len = len(chars)

    char2int = {ch:i for i,ch in enumerate(chars)}
    int2char = {i:ch for i,ch in enumerate(chars)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #selecting device


def softmax(x):
    '''
    Compute softmax values for each sets of scores in x.
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    

def decodeBasedOnTopK(one_hot_pred, top_k=1):
    '''
    Selects an integer from the topk distribution.
    '''
    choices = np.argpartition(one_hot_pred, -top_k)[-top_k:]
    prob = one_hot_pred[choices]
    prob = softmax(prob)
    prob = prob / np.sum(prob)
    choice = np.random.choice(choices, 1, p=prob)
    return choice[0]


def sample(model, initial_string, num_chars, topk=5, device='cpu'):
    '''
    Generates sample text.
    '''
    model.to(device)
    model.eval()
    state = model.init_hidden(1)

    #getting hidden states for starter string
    for char in initial_string:
        one_hot = torch.zeros((1, 1, char_len), dtype=torch.float, device=device)
        one_hot[0,0,char2int[char]] = 1
        _, state = model(one_hot, state)
    
    predict_letters = initial_string[-1]

    for i in range(num_chars):
        one_hot = torch.zeros((1, 1, char_len), dtype=torch.float, device=device)
        one_hot[0,0,char2int[predict_letters[-1]]] = 1
        out, state = model(one_hot, state)
        predict_letters += int2char[decodeBasedOnTopK(out[0].cpu().data.numpy(), top_k=topk)]

    print("{} -- {}".format(initial_string, predict_letters[1:]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Character Level Text Generation')

    req_args = parser.add_argument_group('Required Args')
    req_args.add_argument('-m', '--modelname', type=str, metavar='model_name', dest='model_name', required=True, help='Enter valid model name [rnn, lstm, gru]')

    optional_args = parser.add_argument_group('Optional Args')
    optional_args.add_argument('-i', '--initialstr', type=str, metavar='initial_string', dest='initial_str', default='Avengers Assemble', help='Starter string for the model(default: Avengers Assemble)')
    optional_args.add_argument('-n', '--charlength', type=int, metavar='output_chars_length', dest='out_char', default=200, help='Number of characters to output(default: 200)')

    args = parser.parse_args()

    if(args.model_name not in ['rnn', 'lstm', 'gru']):
        print("Entered unknown model name")
        print("Should be from [rnn, lstm, gru] (all lower case)")
        exit()

    init()
    print("Working on",device)

    model_path = 'model/'+args.model_name+".pth"
    model = torch.load(model_path)

    sample(model=model, initial_string=args.initial_str, num_chars=args.out_char, device=device)
