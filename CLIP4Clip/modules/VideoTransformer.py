from turtle import position
import torch
import torch.nn as nn


class Positional_Encoding():
      def __init__(self, position, d_model):
            super(Positional_Encoding).__init__()
            self.pos_encoding = self.poistional_encoding(position, d_model)

      def get_angles(self, position, i, d_model):
            d_model = d_model.type(torch.float32)
            angles = 1 / torch.pow(
                  10000, (2*(i//2)) / d_model
            )
            return position*angles
      
      def postional_encoding(self, position, d_model):
            angles_rads = self.get_angles(
                  position= torch.range(position, dtype=torch.float32)[:,torch.unsqueeze(-1)],
                  i = torch.range(d_model, dtype=torch.float32)[torch.unsqueeze(0):],
                  d_model=d_model
            )
            sines = torch.sin(angles_rads[:, 0::2])
            cosine = torch.cos(angles_rads[:, 1::2])

            angle_rads = torch.zeros(angles_rads.shape)
            angle_rads[:, 0::2] = sines
            angle_rads[:, 1::2] = cosine
            pos_emebedding = torch.tensor(angle_rads)
            pos_emebedding = pos_emebedding[torch.unsqueeze(0),...]
            
            print(pos_emebedding.shape)
            
            value = pos_emebedding.type(torch.float32)

            return value

def  call(self, inputs):
      return inputs * self.post_encoding[:, :input.size()[1],:]


