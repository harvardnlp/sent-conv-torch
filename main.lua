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
cmd:option('-model_type', 'rand', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel')
cmd:option('-data', 'data.hdf5', 'Training data and word2vec data')
cmd:option('-cudnn', 1, 'Use cudnn and GPUs if set to 1, otherwise set to 0')

trainer.init_cmd(cmd)
model_builder.init_cmd(cmd)
cmd:text()

-- parse arguments
local opts = cmd:parse(arg)

-- Read HDF5 training data
print('loading data...')
local f = hdf5.open(opts.data, 'r')
local train = f:read('train'):all()
local train_label = f:read('train_label'):all()
local w2v = f:read('w2v'):all()
print('data loaded!')

-- build model
local model = model_builder:make_net(w2v, opts)

local criterion = nn.ClassNLLCriterion()

-- move to GPU
if opts.cudnn == 1 then
  require 'cutorch'
  model:cuda()
  criterion:cuda()
end

-- get layers
local linear = model_builder:get_linear()
local w2v = model_builder:get_w2v()
local layers = {linear = linear, w2v = w2v}

-- Currently only adadelta allowed
local optim_method = optim.adadelta
if opts.optim_method == 'adadelta' then
  optim_method = optim.adadelta
end

-- Training loop.
for epoch = 1, opts.num_epochs do
  -- shuffle data
  local shuffle = torch.randperm(train:size(1))
  train = train:index(1, shuffle:long())
  train_label = train_label:index(1, shuffle:long())

  print('==> training epoch ' .. epoch)
  trainer:train(train, train_label, model, criterion, optim_method, layers, opts)

  print('==> evaluate...')
  trainer:test(train, train_label, model, criterion, opts)

  print('\n')
end
