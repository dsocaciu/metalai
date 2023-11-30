
import tiktoken

def stoi(chars): 
    return { ch:i for i,ch in enumerate(chars)}

def itos(chars): 
    return { i:ch for i,ch in enumerate(chars)}

def encode(encode_dict, string_to_encode):
    
    return [encode_dict[c] for c in string_to_encode]

def decode(decode_dict, string_to_decode):
    
    return "".join([decode_dict[i] for i in string_to_decode])


def titoken_gpt2():
    tt_enc = tiktoken.get_encoding("gpt2")

    return tt_enc 