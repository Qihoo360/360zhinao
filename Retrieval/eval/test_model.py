from typing import cast, List, Dict, Union
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('qihoo360/360Zhinao-search')  
model = AutoModel.from_pretrained('qihoo360/360Zhinao-search')
sentences = ['天空是什么颜色的', '天空是蓝色的']
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

if __name__ == "__main__":

    with torch.no_grad():
        last_hidden_state = model(**inputs, return_dict=True).last_hidden_state
        embeddings = last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.cpu().numpy()

    print("embeddings:")
    print(embeddings)

    cos_sim = np.dot(embeddings[0], embeddings[1]) 
    print("cos_sim:", cos_sim)



