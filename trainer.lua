require 'torch'
require 'nn'

local trainer = torch.class('trainer')

function trainer:__init()
  self.model = require 'model'
  self.train_data, self.test_data = require 'data'
end

function trainer:train()
  print('==> training net on sample...')
  self.model:training()
  -- pass
end

function trainer:evaluate_sentence(sentence)
  print('==> testing net on sample...')
  self.model:evaluate()
  self.model:forward(sentence)
end

return trainer

