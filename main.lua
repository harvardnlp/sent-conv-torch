require 'hdf5'
require 'nn'
local trainer = require 'trainer'
local model = require 'model'

local f = hdf5.open('rt-polarity.hdf5', 'r')
local train = f:read('train'):all()
local train_label = f:read('train_label'):all()
-- local word_idx_map = f:read('word_idx_map'):all()

-- temporary
local test = train:clone()
local test_label = train_label:clone()

-- Format is
-- *  'data': 2-d tensor, train size by max sentence len
-- *  'label': 1-d tensor, train size
local train_data = {
  data = train,
  labels = train_label
}
local test_data = {
  data = test,
  labels = test_label
}

local train = trainer.new()
local criterion = nn.ClassNLLCriterion()

for i = 1, 5 do
  train:train(train_data, model, criterion)
end

train:test(test_data)
