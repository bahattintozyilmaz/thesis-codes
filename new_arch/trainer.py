import torch
from torch import optim
from torch import nn
from dataset import device, WordConverter

torch.manual_seed(1337)

teacher_forcing_ratio = 1
MAX_LENGTH = 40


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion, max_length=MAX_LENGTH, use_teacher_forcing=True, train_this_batch=True):
    if train_this_batch:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    target_length = target_tensor.size(0)
    loss = 0
    outs = []

    encoder_output = encoder(input_tensor)
    decoder_input = torch.tensor([[WordConverter.SOS]], device=device)
    decoder_hidden = (encoder_output, torch.zeros(encoder_output.size()))
    decoder_hiddens = [decoder_hidden]

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_hiddens.append(decoder_hidden)
            topv, topi = decoder_output.topk(1)
            outs.append(topi.squeeze().detach())
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == WordConverter.EOS:
                break

    loss.backward()
    #print(target_tensor, outs)
    #for i, h in enumerate(decoder_hiddens):
    #    print(i, h)
    '''for name, param in encoder.named_parameters(prefix='encoder'):
        try:
            print(name, param.data.norm(), param.grad.norm())
        except:
            pass
    for name, param in decoder.named_parameters(prefix='decoder'):
        print(name, param.data.norm(), param.grad.norm())'''

    if train_this_batch:
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / target_length


def train_epoch(data, encoder, decoder, print_every=10, train_every=1, total=1, learning_rate=0.01):
    print_loss_total = 0  # Reset every print_every
    avgs = []
    encoder.train()
    decoder.train()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.015)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.005)
    criterion = nn.NLLLoss()

    for i, data in enumerate(data):
        training_pair = data
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, train_this_batch=(i % train_every == train_every-1))
        print_loss_total += loss

        if i % print_every == print_every-1:
            print_loss_avg = print_loss_total / print_every
            avgs.append(print_loss_avg)
            # avgs = avgs[-print_every:]
            print_loss_total = 0
            print('(%d %d%%) %.4f, running %.4f' % (i, i / total * 100, print_loss_avg, sum(avgs)/len(avgs)))

def evaluate(input_tensor, target_tensor, encoder, decoder, criterion):
    target_length = target_tensor.size(0)
    loss = 0

    encoder_output = encoder(input_tensor)
    decoder_input = torch.tensor([[WordConverter.SOS]], device=device)
    decoder_hidden = encoder_output

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == WordConverter.EOS:
            break

    return loss.item() / target_length

def run_sample(sample, encoder, decoder):
    target_length = sample[1].size(0)
    encoder_output = encoder(sample[0])
    decoder_input = torch.tensor([[WordConverter.SOS]], device=device)
    decoder_hidden = (encoder_output, torch.zeros(encoder_output.size()))
    print(decoder_hidden)
    result = []
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        result.append(decoder_input)
        if decoder_input.item() == WordConverter.EOS:
            break
    print(result)
    return result