import torch
import pprint
from BitePairEncoding import bite_pair_encoding, encode_pair_bite, decode_pair_bite

#Tokenizer class using bite paire encoding algorithm
class BasicTextTokenizer():
  def __init__(self,text, num_reduction) -> None:
    list_token = sorted(set(text))
    self.to_encode_ct = {token: id for id, token in enumerate(list_token)}
    self.to_decode_ct = {id: token for id, token in enumerate(list_token)}
    print("---Bite Pair Encoding---")
    self.reduction(text, num_reduction)
    self.size = len(self.to_decode_ct) + num_reduction

  #encode text numerically them apply the bite pair encoding algorithm and store the new reduction rules  
  def reduction(self, text, num_reduction):
    base_tokenized_text = [self.to_encode_ct[l] for l in text]
    total_token = len(self.to_decode_ct)
    _, self.reduction_rules = bite_pair_encoding(base_tokenized_text , num_reduction, total_token)

  #encode and return the input text
  def encode(self, text: str) -> torch.Tensor:
    tokenized_text = encode_pair_bite([self.to_encode_ct[l] for l in text], self.reduction_rules)
    return torch.tensor(tokenized_text, dtype=torch.long)

  #encode and return the input text
  def decode(self, tokens: torch.Tensor) -> str:
    tokens = tokens.tolist()
    
    tokens = decode_pair_bite(tokens, self.reduction_rules)
    return "".join([self.to_decode_ct[token] for token in tokens])

  #return the vocab size of the tokenizer including token created by the bite pair encoding
  def vocab_size(self) -> int:
    return self.size   


if __name__ == "__main__":
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()[:500]
    
    tokenizer = BasicTextTokenizer(text,10)
    #print(f"Tokenizer:\n{pprint.pformat(tokenizer.to_decode)}")

    tokenized_text = tokenizer.encode(text)
    print(tokenized_text)
    print(tokenizer.reduction_rules)
    decode_text = tokenizer.decode(tokenized_text)
    print(decode_text==text)
    
    