import collections
from tqdm import tqdm



def bite_pair_encoding(tokenized_text, num_reduction, total_token):
  tokenized_text = tokenized_text.copy()
  reduction_rules = []
  for i in tqdm(range(num_reduction)):
    reduction_rules.append((max_pair(tokenized_text), total_token))
    total_token += 1
    target_rule = reduction_rules[-1]
    tokenized_text = apply_rule(tokenized_text, target_rule)
  return tokenized_text, reduction_rules

def max_pair(tokenized_text):
  pairs = {}
  for pair in zip(tokenized_text[:-1],tokenized_text[1:]):
    pairs[pair] = pairs.get(pair, 0) + 1
  max_pair = max(pairs, key=pairs.get)
  return max_pair

def apply_rule(tokenized_text, target_rule):
  for i in range(len(tokenized_text)-1):
    if (tokenized_text[i], tokenized_text[i+1]) == target_rule[0]:
      tokenized_text[i] = target_rule[1]
      tokenized_text[i+1] = float('inf')
  
  return list(filter(lambda x : x != float('inf'), tokenized_text))

def unapply_rule(tokenized_text, target_rule):
  i=0
  result_text=[]
  for token in tokenized_text:
    if token == target_rule[1]:
      result_text.extend(target_rule[0])
    else:
      result_text.append(token)
      
  return result_text

def encode_pair_bite(tokenized_text, reduction_rules):
  tokenized_text = tokenized_text.copy()
  reduction_rules = sorted(reduction_rules,key= lambda x: max(x[0][0], x[0][1]))
  for rule in reduction_rules:
    tokenized_text = apply_rule(tokenized_text, rule)
  return tokenized_text

def decode_pair_bite(tokenized_text, reduction_rules):
  reduction_rules = sorted(reduction_rules,key= lambda x: x[1], reverse=True)
  for rule in reduction_rules:
    tokenized_text = unapply_rule(tokenized_text, rule)
  return tokenized_text


if __name__ == "__main__":
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    
    list_token = sorted(set(text))
    to_encode_ct = {token: id for id, token in enumerate(list_token)}
    base_tokenized_text = [to_encode_ct[l] for l in text]
    
    base_tokenized_text  = base_tokenized_text [:500]
    total_token = len(to_encode_ct)
    num_reduction = 10
    
    
    tokenized_text, reduction_rules = bite_pair_encoding(base_tokenized_text , num_reduction, total_token)
    tokenized_text = encode_pair_bite(base_tokenized_text, reduction_rules)
    tokenized_text_decode = decode_pair_bite(tokenized_text, reduction_rules)
    
    print(tokenized_text_decode == base_tokenized_text)