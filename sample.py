import torch
import argparse

from models import RNNModel, LSTMModel, GRUModel
from utils import sample


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #selecting device
    print("Working on",device)

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

    model_path = 'model/'+args.model_name+".pth"
    model = torch.load(model_path)

    sample(model=model, initial_letters=args.initial_str, num_letters=args.out_char, device=device)
