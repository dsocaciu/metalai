import os
import constants
from extract_lyrics import extractLyrics
import encoders
import torch
import torch.nn as nn
from torch.nn import functional as F


# super simple bigram model - from andrej karpathy
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # token embedding table is of size vocab_size by vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.lm_head = nn.Linear(n_embed,vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        
        #each idx will grab the idx-th row of the token embedding table 
        logits = self.token_embedding_table(idx) # (B,T,C) batch = 4, time = 8 , vocab = 97
        #logits are scores for next character in a sequence 
        logits = self.lm_head(tok_emb)


        #reshape the logits
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #2 dimensional array instead of 3 ( B by T by C )
            targets = targets.view(B*T) #1 dimensional array instead of 2 (B by T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def main():

    max_iters = 3000
    eval_interval = 300
    learning_rate = 1e-2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    

    lyrics = extractLyrics(constants.CENTRAL_METALLICA_LYRICS)
    
    directory = os.getcwd()
    file_name = "metallica_lyrics.txt"
    metallica_file = f"{directory}/{file_name}"

    if not os.path.exists(metallica_file):

        lyrics.write_metallica_txt(metallica_file)

    
    with open(metallica_file,'r', encoding='utf-8') as f:
        text = f.read()

    #print("length of the metallica lyrics: ", len(text))

    #first 1000 characters
    #print(text[:1000])

    #figure out the unique characters

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    #print("".join(chars))
    #print(vocab_size)

    encode_dict = encoders.stoi(chars)
    decode_dict = encoders.itos(chars)




    print(encoders.decode(decode_dict,encoders.encode(encode_dict,"Hello@")))
    
    assert encoders.decode(decode_dict,encoders.encode(encode_dict,"Hello@")) == "Hello@"

    tt_gpt = encoders.titoken_gpt2()

    #print(tt_gpt.n_vocab)

    assert tt_gpt.decode(tt_gpt.encode("Metallica")) == "Metallica"


    data = torch.tensor(encoders.encode(encode_dict,text), dtype=torch.long)

    print(data.shape,data.dtype)

    print(data[:1000])


    #split for training and validation sets
    #train on the first 90%
    n = int(.9*len(data))
    train_data = data[:n]
    val_data = data[:n]

    #train on random chunks of data

    block_size = 8
    print(train_data[:block_size+1])

    x = train_data[:block_size]
    y = train_data[1:block_size+1]

    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} the target is {target}")


    # data loading - https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py
    def get_batch(split):
        
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,)) #random offsets into the training set
        x = torch.stack([data[i:i+block_size] for i in ix]) #1 dimensional tensor
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #offset by 1
        x, y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    torch.manual_seed(1988)
    batch_size = 4 #independent sequences
    block_size = 8 # maximum context length for predictions

    xb,yb = get_batch("train")
    print(f"inputs -> shape:{xb.shape}\nxb:{xb}")

    print(f"targets -> shape:{yb.shape}\nyb:{yb}")


    print("------")

    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b,t]
            print(f"when input is {context.tolist()} the target is {target}")


    print(xb)

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    logits,loss = m(xb,yb)
    print(logits.shape)
    print(loss)

    idx = torch.zeros((1,1), dtype=torch.long) #the index is 1 by 1 tensor of integer type
    decode = encoders.decode
    
    #print(decode(decode_dict,m.generate( idx , max_new_tokens=100)[0].tolist()))
    #first totally random out -> vfODHU5FkS]a)(Q@EAt@oXp©5m©âGz3E-TNS%
    
    
    #now train the model

    #INITIAL TRAINING
    # optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3) #learning rate

    # batch_size = 32
    # for steps in range(10000):

    #     #sample a batch of data
    #     xb, yb = get_batch("train")

    #     #evaluate the loss
    #     logits, loss = m(xb,yb)
    #     optimizer.zero_grad(set_to_none=True)
    #     loss.backward()
    #     optimizer.step()

    # print(loss.item())

    #print(decode(decode_dict,m.generate( idx , max_new_tokens=100)[0].tolist()))

    #training output
    # YROMumefliseve auer
    # Me war I dallle the dl
    # WXY t I ppy o t tyrse g ng Yourn


    # hinde lion Yo breront

    ####UPDATED TRAINING

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)
    #print(decode(decode_dict,m.generate(context, max_new_tokens=500)[0].tolist()))


    #self-attention mathematical seed
    torch.manual_seed(1337)
    B,T,C = 4,8,2 #batch, time, channels

    x = torch.randn(B,T,C)
    print(x.shape)

    xbow = torch.zeros((B,T,C)) #bag of words ->averaging words
    for b in range(B): #iterate over batch dimensions
        for t in range(T): #iterative over  time 
            xprev = x[b,:t+1] # (t,C)
            xbow[b,t] = torch.mean(xprev,0) 

    print(x[0])

    print(xbow[0])

    #the above is inefficient 

    #use matrix multiplication to speed things up 

    wei = torch.tril(torch.ones(T,T))
    wei = wei / wei.sum(1,keepdim=True)
    print(wei)

    xbow2 = wei @ x # (B,T,T) @ (B,T,C) ---> (B,T,C)

    print(xbow2)
    print(torch.allclose(xbow,xbow2))


    #use softmax
    tril = torch.tril(torch.ones(T,T))
    wei = torch.zeros((T,T))
    wei = wei.masked_fill(tril == 0, float('-inf')) #tokens from the past cannot communicate
    wei = F.softmax(wei, dim=-1) #normalization operation
    xbow3 = wei @ x 
    torch.allclose(xbow,xbow3)

    print(wei)


if __name__ == '__main__':
    main()