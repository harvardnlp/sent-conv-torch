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
cmd:option('-use_cv', 1, 'If hdf5 data is not already split train/dev/test, we use CV to split')

trainer.init_cmd(cmd)
model_builder.init_cmd(cmd)
cmd:text()

-- parse arguments
local opts = cmd:parse(arg)
--torch.manualSeed(opts.seed)

-- Currently only adadelta allowed
local optim_method
if opts.optim_method == 'adadelta' then
  --optim_method = optim.adadelta
  optim_method = Trainer.adadelta
elseif opts.optim_method == 'adam' then
  optim_method = optim.adam
end

-- Read HDF5 training data
print('loading data...')
local f = hdf5.open(opts.data, 'r')
local data = f:read('data'):all()
local data_label = f:read('data_label'):all()
local w2v = f:read('w2v'):all()
print('data loaded!')

opts.vocab_size = w2v:size(1)
print('vocab size: ' .. opts.vocab_size)

-- Zero-pad at start
local max_filt_sz = torch.max(torch.Tensor{opts.kernel1, opts.kernel2, opts.kernel3})
local tmp_data = torch.ones(data:size(1), data:size(2) + max_filt_sz - 1)
tmp_data[{{}, {max_filt_sz,tmp_data:size(2)}}]:copy(data)

if opts.learn_start == 1 then
  -- add start padding that gets learned
  tmp_data[{{}, {1,max_filt_sz-1}}] = opts.vocab_size + 1
  opts.vocab_size = opts.vocab_size + 1
  w2v = torch.cat(w2v, torch.Tensor(1, w2v:size(2)):uniform(-0.25, 0.25), 1)-- append to w2v
end
data = tmp_data
collectgarbage()

opts.max_sent = data:size(2)
print('data size:')
print(data:size())

local N = data:size(1)
local fold_dev_scores = {}
local fold_test_scores = {}

-- shuffle data
local shuffle = torch.randperm(data:size(1)):long()
data = data:index(1, shuffle)
data_label = data_label:index(1, shuffle)

for fold = 1, opts.folds do
  local fold_time = sys.clock()

  print()
  print('==> fold ' .. fold)

  local test, test_label
  local train, train_label
  local dev, dev_label
  if opts.use_cv == 1 then
    -- make train/dev/test data (90/10 split for train/test)
    local i_start = math.floor((fold - 1) * 0.1 * N + 1)
    local i_end = math.floor(fold * 0.1 * N)
    test = data:narrow(1, i_start, i_end - i_start + 1)
    test_label = data_label:narrow(1, i_start, i_end - i_start + 1)
    train = torch.cat(data:narrow(1, 1, i_start), data:narrow(1, i_end, N - i_end + 1), 1)
    train_label = torch.cat(data_label:narrow(1, 1, i_start), data_label:narrow(1, i_end, N - i_end + 1), 1)

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
  else
    print('hi')
    -- Assume data is already split for us.
  end

  if opts.debug == 1 then
    print('train size:')
    print(train:size())
    print('dev size:')
    print(dev:size())
  end

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
  local skip_conv = model_builder:get_layer(model, 'skip_conv')
  print(skip_conv)
  local layers = {linear = linear, w2v = w2v_layer, skip_conv = skip_conv}

  -- Training loop.
  local best_model = model:clone()
  local best_epoch = 1
  local best_err = 0.0

  for epoch = 1, opts.num_epochs do
    local epoch_time = sys.clock()

    local train_err = trainer:train(train, train_label, model, criterion, optim_method, layers, opts)

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
torch.save(savefile, { dev_scores = fold_dev_scores, test_scores = fold_test_scores, opts = opts })
