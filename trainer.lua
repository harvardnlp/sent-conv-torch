require 'torch'
require 'nn'

local trainer = torch.class('trainer')

function trainer:__init()
  self.var = 10
end

function trainer:train()
  self.var = self.var + 1
end

return trainer

