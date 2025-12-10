from importlib.metadata import version
import urllib.request
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

print(f'Version of tiktoken is: {version("tiktoken")}')

tokenizer = tiktoken.get_encoding("gpt2")

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt,  batch_size=4, max_length=256, stride=128, shuffle=True, drop_last = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

with open("../../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
# print(f"First batch {first_batch}")
second_batch = next(data_iter)
# print(f"Second batch {second_batch}")

# Why does batch size matter?
# A: Determines how many samples you process at the same time in one forward pass.
# Max length and stride determine how many examples the dataset has and how much they overlap


torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(num_embeddings=6, embedding_dim = 3)
# print(embedding_layer.weight)

# Encoding positions
"""
OpenAI uses absolute positional embeddings that are optimized during the training process rather than being fixed or predefined like the positional encodings of the original transformer. This optimization is part of the model training itself.

Pretty neat, useful to know where in a sequence a word with the same embedding is as it can be useful for subject/object understanding e..g the blue fox jumps over the fox

Important to know that the first and second foxes are distinct entities, which wouldn't be captured by regular token embedding alone.
"""

token_embedding_layer = torch.nn.Embedding(num_embeddings=50257, embedding_dim=256)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(f"Token IDs: {inputs}")
print(f"Inputs shape: {inputs.shape}")

input_token_embeddings = token_embedding_layer(inputs)
print(input_token_embeddings.shape)

'''
Inputs shape: torch.Size([8, 4]) <- 8 examples in a batch, 4 tokens per example

torch.Size([8, 4, 256]) <- 8 examples in a batch, 4 tokens per example, each token is sent into 256 dimensional space
You could view the comma of a tensor as "for each"
'''

# Now we create the positional embeddings
context_length = max_length
positional_embedding_layer = torch.nn.Embedding(context_length, embedding_dim=256)
positional_embeddings = positional_embedding_layer(torch.arange(context_length))
print(positional_embeddings.shape)
# torch.Size([4, 256])

input_embeddings = input_token_embeddings + positional_embeddings




