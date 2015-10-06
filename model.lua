require 'torch'
require 'nn'
require 'fbcunn'
require 'cunn'

-- Hyperparameters
local num_channels = 1
local num_feat_maps = 100 -- Number of feature maps after 1st convolution.
local filt_size = 3
local num_classes = 2
local dropout_p = 0.5

local vocab_size = 21421 -- TODO(jeffreyling): find vocab size of word2vec
local vec_size = 300 -- word2vec vector size

local model = nn.Sequential()
model:add(nn.LookupTable(vocab_size, vec_size)) -- LookupTable for word2vec
model:add(nn.TemporalConvolution(vec_size, num_feat_maps, filt_size))
model:add(nn.ReLU())
model:add(nn.Transpose({2,3})) -- swap feature maps and time
model:add(nn.Max(3)) -- max over time

model:add(nn.Dropout(dropout_p))
model:add(nn.Linear(num_feat_maps, num_classes))
model:add(nn.LogSoftMax())

return model
