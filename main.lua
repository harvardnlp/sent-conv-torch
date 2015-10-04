require 'cutorch'
require 'hdf5'
require 'nn'
local Trainer = require 'trainer'
local model = require 'model'

local f = hdf5.open('rt-polarity.hdf5', 'r')
local train = f:read('train'):all()
local train_label = f:read('train_label'):all()
-- local word_idx_map = f:read('word_idx_map'):all()

local trainer = Trainer.new()
local criterion = nn.ClassNLLCriterion()

-- move to GPU
model:cuda()
criterion:cuda()

for epoch = 1, 10 do
  -- shuffle data
  local shuffle = torch.randperm(train:size(1))
  train = train:index(1, shuffle:long())
  train_label = train_label:index(1, shuffle:long())

  print('==> training epoch ' .. epoch)
  trainer:train(train, train_label, model, criterion)

  print('==> evaluate...')
  trainer:test(train, train_label, model, criterion)
end
