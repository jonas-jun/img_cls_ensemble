import torch
import torch.nn as nn
import clip

class CLIPextractor(nn.Module):
    def __init__(self, pretrain_name='RN50x16'):
        super(CLIPextractor, self).__init__()
        model, preprocess = clip.load(pretrain_name)
        self.clip = model.visual
        self.input_resolution = model.visual.input_resolution
        print(pretrain_name)
        print("model.visual.input_resolution:", model.visual.input_resolution)
        print("model.visual.output_dim      :", model.visual.output_dim)
    
    def forward(self, input):
        return self.clip(input)