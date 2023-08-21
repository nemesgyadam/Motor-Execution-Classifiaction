import torch
from transformer_block import BasicTransformerBlock


block = BasicTransformerBlock(dim=64, num_attention_heads=8, attention_head_dim=128, dropout=.5,
                              cross_attention_dim=256)

eeg = torch.zeros((4, 8, 256))
subj_emb = torch.zeros((4, 1, 64))

y = block(hidden_states=subj_emb, encoder_hidden_states=eeg)

print(y.shape)


torch.nn.LazyConv2d()
torch.nn.LazyBatchNorm2d()

