require 'hdf5'
require 'nn'
require 'optim'
require 'lfs'

-- Flags
cmd = torch.CmdLine()

cmd:text()
cmd:text()
cmd:text('Convolutional net for sentence classification')
cmd:text()
cmd:text('Options')
cmd:option('-model_type', 'rand', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')
cmd:option('-data', 'MR.hdf5', 'Training data and word2vec data')
cmd:option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
cmd:option('-seed', -1, 'random seed, set -1 for actual random')
cmd:option('-folds', 10, 'number of folds to use. If test set provided, folds=1. max 10')
cmd:option('-debug', 0, 'print debugging info including timing, confusions')
cmd:option('-gpuid', 1, 'GPU device id to use.')
cmd:option('-savefile', '', 'Name of output file, which will hold the trained model, model parameters, and training scores. Default filename is TIMESTAMP_results')
cmd:option('-zero_indexing', 0, 'If data is zero indexed')
cmd:text()

-- Preset by preprocessed data
cmd:option('-has_test', 1, 'If data has test, we use it. Otherwise, we use CV on folds')
cmd:option('-has_dev', 1, 'If data has dev, we use it, otherwise we split from train')
cmd:option('-num_classes', 2, 'Number of output classes')
cmd:option('-max_sent', 59, 'maximum sentence length')
cmd:option('-vec_size', 300, 'word2vec vector size')
cmd:option('-vocab_size', 18766, 'Vocab size')
cmd:text()

-- Training own dataset
cmd:option('-train_only', 0, 'Set to 1 to only train on data. Default is cross-validation')
cmd:option('-test_only', 0, 'Set to 1 to only do testing. Must have a -warm_start_model')
cmd:option('-warm_start_model', '', 'Path to .t7 file with pre-trained model. Should contain a table with key \'model\'')
cmd:text()

-- Training hyperparameters
cmd:option('-num_epochs', 25, 'Number of training epochs')
cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
cmd:option('-L2s', 3, 'L2 normalize weights')
cmd:option('-batch_size', 50, 'Batch size for training')
cmd:text()

-- Model hyperparameters
cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
cmd:option('-kernels', '{3,4,5}', 'Kernel sizes of convolutions, table format.')
cmd:option('-skip_kernel', 0, 'Use skip kernel')
cmd:option('-dropout_p', 0.5, 'p for dropout')
cmd:option('-highway_mlp', 0, 'Number of highway MLP layers')
cmd:option('-highway_conv_layers', 0, 'Number of highway MLP layers')
cmd:text()

function get_layer(model, name)
  local named_layer
  function get(layer)
    if torch.typename(layer) == name or layer.name == name then
      named_layer = layer
    end
  end

  model:apply(get)
  return named_layer
end

-- build model for training
function build_model(w2v)
  local ModelBuilder = require 'model.convNN'
  local model_builder = ModelBuilder.new()

  local model
  if opt.warm_start_model == '' then
    model = model_builder:make_net(w2v)
  else
    require "nngraph"
    if opt.cudnn == 1 then
      require "cudnn"
      require "cunn"
    end
    model = torch.load(opt.warm_start_model).model
  end

  local criterion = nn.ClassNLLCriterion()

  -- move to GPU
  if opt.cudnn == 1 then
    model = model:cuda()
    criterion = criterion:cuda()
  end

  -- get layers
  local layers = {}
  layers['linear'] = get_layer(model, 'nn.Linear')
  layers['w2v'] = get_layer(model, 'nn.LookupTable')
  if opt.skip_kernel > 0 then
    layers['skip_conv'] = get_layer(model, 'skip_conv')
  end
  if opt.model_type == 'multichannel' then
    layers['chan1'] = get_layer(model, 'channel1')
  end

  return model, criterion, layers
end

function train_loop(all_train, all_train_label, test, test_label, dev, dev_label, w2v)
  -- Initialize objects
  local Trainer = require 'trainer'
  local trainer = Trainer.new()

  local optim_method
  if opt.optim_method == 'adadelta' then
    optim_method = optim.adadelta
  elseif opt.optim_method == 'adam' then
    optim_method = optim.adam
  end

  local best_model -- save best model
  local fold_dev_scores = {}
  local fold_test_scores = {}

  local train, train_label -- training set for each fold
  if opt.has_test == 1 then
    train = all_train
    train_label = all_train_label
  end

  -- Training folds.
  for fold = 1, opt.folds do
    local timer = torch.Timer()
    local fold_time = timer:time().real

    print()
    print('==> fold ', fold)

    if opt.has_test == 0 and opt.train_only == 0 then
      -- make train/test data (90/10 split for train/test)
      local N = all_train:size(1)
      local i_start = math.floor((fold - 1) * (N / opt.folds) + 1)
      local i_end = math.floor(fold * (N / opt.folds))
      test = all_train:narrow(1, i_start, i_end - i_start + 1)
      test_label = all_train_label:narrow(1, i_start, i_end - i_start + 1)
      train = torch.cat(all_train:narrow(1, 1, i_start), all_train:narrow(1, i_end, N - i_end + 1), 1)
      train_label = torch.cat(all_train_label:narrow(1, 1, i_start), all_train_label:narrow(1, i_end, N - i_end + 1), 1)
    end

    if opt.has_dev == 0 then
      -- shuffle train to get dev/train split (10% to dev)
      -- We organize our data in batches at this split before epoch training.
      local J = train:size(1)
      local shuffle = torch.randperm(J):long()
      train = train:index(1, shuffle)
      train_label = train_label:index(1, shuffle)

      local num_batches = math.floor(J / opt.batch_size)
      local num_train_batches = torch.round(num_batches * 0.9)

      local train_size = num_train_batches * opt.batch_size
      local dev_size = J - train_size
      dev = train:narrow(1, train_size+1, dev_size)
      dev_label = train_label:narrow(1, train_size+1, dev_size)
      train = train:narrow(1, 1, train_size)
      train_label = train_label:narrow(1, 1, train_size)
    end

    -- build model
    local model, criterion, layers = build_model(w2v)

    -- Call getParameters once
    local params, grads = model:getParameters()

    -- Training loop.
    best_model = model:clone()
    local best_epoch = 1
    local best_err = 0.0

    -- Training.
    -- Gradient descent state should persist over epochs
    local state = {}
    for epoch = 1, opt.num_epochs do
      local epoch_time = timer:time().real

      -- Train
      local train_err = trainer:train(train, train_label, model, criterion, optim_method, layers, state, params, grads)
      -- Dev
      local dev_err = trainer:test(dev, dev_label, model, criterion)
      if dev_err > best_err then
        best_model = model:clone()
        best_epoch = epoch
        best_err = dev_err 
      end

      if opt.debug == 1 then
        print()
        print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
        print('\n')
      end

      print('epoch:', epoch, 'train perf:', 100*train_err, '%, val perf ', 100*dev_err, '%')
    end

    print('best dev err:', 100*best_err, '%, epoch ', best_epoch)
    table.insert(fold_dev_scores, best_err)

    -- Testing.
    if opt.train_only == 0 then
      local test_err = trainer:test(test, test_label, best_model, criterion)
      print('test perf ', 100*test_err, '%')
      table.insert(fold_test_scores, test_err)
    end

    if opt.debug == 1 then
      print()
      print('time for one fold: ', (timer:time().real - fold_time * 1000), 'ms')
      print('\n')
    end
  end

  return fold_dev_scores, fold_test_scores, best_model
end

function load_data()
  local train, train_label
  local dev, dev_label
  local test, test_label

  print('loading data...')
  local f = hdf5.open(opt.data, 'r')
  local w2v = f:read('w2v'):all()
  train = f:read('train'):all()
  train_label = f:read('train_label'):all()
  opt.num_classes = torch.max(train_label)

  if f:read('dev'):dataspaceSize()[1] == 0 then
    opt.has_dev = 0
  else
    opt.has_dev = 1
    dev = f:read('dev'):all()
    dev_label = f:read('dev_label'):all()
  end
  if f:read('test'):dataspaceSize()[1] == 0 then
    opt.has_test = 0
  else
    opt.has_test = 1
    test = f:read('test'):all()
    test_label = f:read('test_label'):all()
  end
  print('data loaded!')

  return train, train_label, test, test_label, dev, dev_label, w2v
end

function main()
  -- parse arguments
  opt = cmd:parse(arg)

  if opt.seed ~= -1 then
    torch.manualSeed(opt.seed)
  end
  if opt.cudnn == 1 then
    require 'cutorch'
    if opt.seed ~= -1 then
      cutorch.manualSeedAll(opt.seed)
    end
    cutorch.setDevice(opt.gpuid)
  end

  -- Read HDF5 training data
  local train, train_label
  local test, test_label
  local dev, dev_label
  local w2v
  train, train_label, test, test_label, dev, dev_label, w2v = load_data()

  opt.vocab_size = w2v:size(1)
  opt.vec_size = w2v:size(2)
  opt.max_sent = train:size(2)
  print('vocab size: ', opt.vocab_size)
  print('vec size: ', opt.vec_size)

  -- Retrieve kernels
  loadstring("opt.kernels = " .. opt.kernels)()

  if opt.zero_indexing == 1 then
    train:add(1)
    train_label:add(1)
    if dev ~= nil then
      dev:add(1)
      dev_label:add(1)
    end
    if test ~= nil then
      test:add(1)
      test_label:add(1)
    end
  end

  if opt.test_only == 1 then
    assert(opt.warm_start_model ~= '', 'must have -warm_start_model for testing')
    assert(opt.has_test == 1)
    local Trainer = require "trainer"
    local trainer = Trainer.new()
    print('Testing...')
    local model, criterion = build_model(w2v)
    local test_err = trainer:test(test, test_label, model, criterion)
    print('Test score:', test_err)
    os.exit()
  end

  if opt.has_test == 1 or opt.train_only == 1 then
    -- don't do CV if we have a test set, or are training only
    opt.folds = 1
  end

  -- training loop
  local fold_dev_scores, fold_test_scores, best_model = train_loop(train, train_label, test, test_label, dev, dev_label, w2v)

  print('dev scores:')
  print(fold_dev_scores)
  print('average dev score: ', torch.Tensor(fold_dev_scores):mean())

  if opt.train_only == 0 then
    print('test scores:')
    print(fold_test_scores)
    print('average test score: ', torch.Tensor(fold_test_scores):mean())
  end

  -- make sure output directory exists
  if not path.exists('results') then lfs.mkdir('results') end

  local savefile
  if opt.savefile ~= '' then
    savefile = opt.savefile
  else
    savefile = string.format('results/%s_model.t7', os.date('%Y%m%d_%H%M'))
  end
  print('saving results to ', savefile)

  local save = {}
  save['dev_scores'] = fold_dev_scores
  if opt.train_only == 0 then
    save['test_scores'] = fold_test_scores
  end
  save['opt'] = opt
  save['model'] = best_model
  save['embeddings'] = get_layer(best_model, 'nn.LookupTable').weight
  torch.save(savefile, save)
end

main()
