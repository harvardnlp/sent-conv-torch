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
cmd:option('-num_epochs', 25, 'Number of training epochs')
cmd:option('-model_type', 'rand', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (TODO)')
cmd:option('-data', 'data.hdf5', 'Training data and word2vec data')
cmd:option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
cmd:option('-seed', 35392, 'random seed')

trainer.init_cmd(cmd)
model_builder.init_cmd(cmd)
cmd:text()

-- parse arguments
local opts = cmd:parse(arg)
torch.manualSeed(opts.seed)

-- Currently only adadelta allowed
local optim_method = optim.adadelta
if opts.optim_method == 'adadelta' then
  optim_method = optim.adadelta
end

-- Read HDF5 training data
print('loading data...')
local f = hdf5.open(opts.data, 'r')
local data = f:read('data'):all()
local data_label = f:read('data_label'):all()
local w2v = f:read('w2v'):all()
print('data loaded!')

-- Zero-pad at start
local max_filt_sz = 3 -- this should depend on filter size
local tmp_data = torch.ones(data:size(1), data:size(2) + max_filt_sz - 1)
tmp_data[{{}, {max_filt_sz,tmp_data:size(2)}}]:copy(data)
data = tmp_data
collectgarbage()

opts.vocab_size = w2v:size(1)
print('vocab size: ' .. opts.vocab_size)

local N = data:size(1)
local fold_dev_scores = {}
local fold_test_scores = {}
for fold = 1, 10 do
  print('==> fold ' .. fold)

  -- make train/dev/test data (90/10 split for train/test)
  local i_start = (fold - 1) * 0.1 * N + 1
  local i_end = fold * 0.1 * N
  local test = data:narrow(1, i_start, i_end - i_start + 1)
  local test_label = data_label:narrow(1, i_start, i_end - i_start + 1)
  local train = torch.cat(data:narrow(1, 1, i_start), data:narrow(1, i_end, N - i_end + 1), 1)
  local train_label = torch.cat(data_label:narrow(1, 1, i_start), data_label:narrow(1, i_end, N - i_end + 1), 1)

  -- shuffle to get dev/train split (10% to dev)
  local J = train:size(1)
  local shuffle = torch.randperm(J):long()
  train = train:index(1, shuffle)
  train_label = train_label:index(1, shuffle)

  local dev_size = 0.1 * J
  local dev = train:narrow(1, 1, dev_size)
  local dev_label = train_label:narrow(1, 1, dev_size)
  local train_size = J - dev_size + 1
  train = train:narrow(1, dev_size, train_size)
  train_label = train_label:narrow(1, dev_size, train_size)

  -- build model
  local model = model_builder:make_net(w2v, opts)

  local criterion = nn.ClassNLLCriterion()

  -- move to GPU
  if opts.cudnn == 1 then
    require 'cutorch'
    cutorch.setDevice(0)
    model:cuda()
    criterion:cuda()
  end

  -- get layers
  local linear = model_builder:get_linear()
  local w2v_layer = model_builder:get_w2v()
  local layers = {linear = linear, w2v = w2v_layer}

  -- Training loop.
  local best_model = model:clone()
  local best_epoch = 1
  local best_err = 0.0

  for epoch = 1, opts.num_epochs do
    -- shuffle data
    shuffle = torch.randperm(train:size(1)):long()
    train = train:index(1, shuffle)
    train_label = train_label:index(1, shuffle)

    print('==> training epoch ' .. epoch)
    trainer:train(train, train_label, model, criterion, optim_method, layers, opts)

    print('\n')
    print('==> evaluate...')
    local err_rate = trainer:test(dev, dev_label, model, criterion, opts)
    if err_rate > best_err then
      best_model = model:clone()
      best_epoch = epoch
      best_err = err_rate
    end

    print('\n')
  end

  print('best epoch: ' .. best_epoch)
  print('best dev err: ' .. 100*best_err .. '%')
  print('\n')
  print('==> test for fold ' .. fold)
  local test_err = trainer:test(test, test_label, best_model, criterion, opts)

  table.insert(fold_dev_scores, best_err)
  table.insert(fold_test_scores, test_err)

  -- reset model
  model_builder.model = nil
end

print('dev scores:')
print(fold_dev_scores)
print('test scores:')
print(fold_test_scores)
