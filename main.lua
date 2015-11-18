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
cmd:option('-data', 'MR.hdf5', 'Training data and word2vec data')
cmd:option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
cmd:option('-seed', 35392, 'random seed')
cmd:option('-folds', 10, 'number of folds to use. max 10')
cmd:option('-learn_start', 0, 'learned start padding')
cmd:option('-debug', 0, 'print debugging info including timing, confusions')

cmd:option('-has_test', 0, 'If data has test, we use it. Otherwise, we use CV on folds')
cmd:option('-has_dev', 0, 'If data has dev, we use it, otherwise we split from train')
cmd:option('-zero_indexing', 0, 'If data is zero indexed')

trainer.init_cmd(cmd)
model_builder.init_cmd(cmd)
cmd:text()

-- parse arguments
local opts = cmd:parse(arg)

local optim_method
if opts.optim_method == 'adadelta' then
  --optim_method = optim.adadelta
  optim_method = optim.adadelta
elseif opts.optim_method == 'adam' then
  optim_method = optim.adam
end

-- Read HDF5 training data
local test, test_label
local train, train_label
local dev, dev_label

print('loading data...')
local f = hdf5.open(opts.data, 'r')
local w2v = f:read('w2v'):all()
if opts.has_dev == 1 then
  test = f:read('test'):all()
  test_label = f:read('test_label'):all()
  train = f:read('train'):all()
  train_label = f:read('train_label'):all()
  dev = f:read('dev'):all()
  dev_label = f:read('dev_label'):all()

  if opts.debug == 1 then
    print(test:size())
    print(train:size())
    print(dev:size())
  end
elseif opts.has_test == 1 then
  test = f:read('test'):all()
  test_label = f:read('test_label'):all()
  train = f:read('train'):all()
  train_label = f:read('train_label'):all()
else
  -- Need CV split
  train = f:read('data'):all()
  train_label = f:read('data_label'):all()
end
print('data loaded!')

opts.vocab_size = w2v:size(1)
opts.vec_size = w2v:size(2)
opts.max_sent = train:size(2)
print('vocab size: ' .. opts.vocab_size)
print('vec size: ' .. opts.vec_size)

if opts.zero_indexing == 1 then
  train:add(1)
  train_label:add(1)
end

if opts.has_test == 1 or opts.has_dev == 1 then
  -- don't do CV if we have a test set
  opts.folds = 1
end

local best_model -- save best model
local fold_dev_scores = {}
local fold_test_scores = {}
for fold = 1, opts.folds do
  local fold_time = sys.clock()

  print()
  print('==> fold ' .. fold)

  if opts.has_test == 0 then
    -- make train/test data (90/10 split for train/test)
    local N = train:size(1)
    local i_start = math.floor((fold - 1) * 0.1 * N + 1)
    local i_end = math.floor(fold * 0.1 * N)
    test = train:narrow(1, i_start, i_end - i_start + 1)
    test_label = train_label:narrow(1, i_start, i_end - i_start + 1)
    train = torch.cat(train:narrow(1, 1, i_start), train:narrow(1, i_end, N - i_end + 1), 1)
    train_label = torch.cat(train_label:narrow(1, 1, i_start), train_label:narrow(1, i_end, N - i_end + 1), 1)
  end

  if opts.has_dev == 0 then
    -- shuffle to get dev/train split (10% to dev)
    -- We organize our data in batches at this split before epoch training.
    local J = train:size(1)
    local shuffle = torch.randperm(J):long()
    train = train:index(1, shuffle)
    train_label = train_label:index(1, shuffle)

    local num_batches = math.floor(J / opts.batch_size)
    local num_train_batches = torch.round(num_batches * 0.9)

    local train_size = num_train_batches * opts.batch_size
    local dev_size = J - train_size
    dev = train:narrow(1, train_size+1, dev_size)
    dev_label = train_label:narrow(1, train_size+1, dev_size)
    train = train:narrow(1, 1, train_size)
    train_label = train_label:narrow(1, 1, train_size)
  end

  print('train size:')
  print(train:size())
  print('dev size:')
  print(dev:size())
  print('test size:')
  print(test:size())

  -- build model
  local model = model_builder:make_net(w2v, opts)

  local criterion = nn.ClassNLLCriterion()

  -- move to GPU
  if opts.cudnn == 1 then
    require 'cutorch'
    --cutorch.setDevice(0)
    model:cuda()
    criterion:cuda()
  end

  -- get layers
  local linear = model_builder:get_layer(model, 'nn.Linear')
  local w2v_layer = model_builder:get_layer(model, 'nn.LookupTable')
  if opts.skip_kernel > 0 then
    local skip_conv = model_builder:get_layer(model, 'skip_conv')
    print(skip_conv)
  end
  local layers = {linear = linear, w2v = w2v_layer, skip_conv = skip_conv}

  -- Training loop.
  best_model = model:clone()
  local best_epoch = 1
  local best_err = 0.0

  -- Gradient descent state should persist over epochs
  local state = {}
  for epoch = 1, opts.num_epochs do
    local epoch_time = sys.clock()

    -- Train
    local train_err = trainer:train(train, train_label, model, criterion, optim_method, layers, state, opts)
    -- Dev
    local dev_err = trainer:test(dev, dev_label, model, criterion, opts)
    if dev_err > best_err then
      best_model = model:clone()
      best_epoch = epoch
      best_err = dev_err 
    end

    if opts.debug == 1 then
      print()
      print('time for one epoch: ' .. ((sys.clock() - epoch_time) * 1000) .. 'ms')
      print('\n')
    end

    print('epoch ' .. epoch .. ', train perf ' .. 100*train_err .. '%, val perf ' .. 100*dev_err .. '%')
  end

  print('best dev err: ' .. 100*best_err .. '%, epoch ' .. best_epoch)
  local test_err = trainer:test(test, test_label, best_model, criterion, opts)
  print('test perf ' .. 100*test_err .. '%')

  table.insert(fold_dev_scores, best_err)
  table.insert(fold_test_scores, test_err)

  if opts.debug == 1 then
    print()
    print('time for one fold: ' .. ((sys.clock() - fold_time) * 1000) .. 'ms')
    print('\n')
  end
end

print('dev scores:')
print(fold_dev_scores)
print('average dev score: ' .. torch.Tensor(fold_dev_scores):mean())
print('test scores:')
print(fold_test_scores)
print('average test score: ' .. torch.Tensor(fold_test_scores):mean())

local savefile = string.format('results/%s_results', os.date('%Y%m%d_%H%M'))
print('saving results to ' .. savefile)
torch.save(savefile, { dev_scores = fold_dev_scores, test_scores = fold_test_scores, opts = opts, model = best_model })
