require 'cutorch'
require 'hdf5'
require 'nn'
require 'optim'
local Trainer = require 'trainer'
local ModelBuilder = require 'model'

-- Initialize objects
local trainer = Trainer.new()
local model_builder = ModelBuilder.new()

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Convolutional net for sentence classification')
cmd:text()
cmd:text('Options')
cmd:option('-num_epochs', 10, 'Number of training epochs')
cmd:option('-model_type', 'static', 'Model type. Options: rand, static, non-static, multichannel')
trainer.init_cmd(cmd)
model_builder.init_cmd(cmd)
cmd:text()

-- parse arguments
local opts = cmd:parse(arg)

-- Read HDF5 training data
local f = hdf5.open('rt-polarity.hdf5', 'r')
local train = f:read('train'):all()
local train_label = f:read('train_label'):all()

local model = model_builder:make_net(opts)
local criterion = nn.ClassNLLCriterion()

-- Currently only adadelta allowed
local optim_method = optim.adadelta
if opts.optim_method == 'adadelta' then
  optim_method = optim.adadelta
end

-- move to GPU
model:cuda()
criterion:cuda()

-- Training loop.
for epoch = 1, opts.num_epochs do
  -- shuffle data
  local shuffle = torch.randperm(train:size(1))
  train = train:index(1, shuffle:long())
  train_label = train_label:index(1, shuffle:long())

  print('==> training epoch ' .. epoch)
  trainer:train(train, train_label, model, criterion, optim_method, opts)

  print('==> evaluate...')
  trainer:test(train, train_label, model, criterion, opts)

  print('\n')
end
