require 'hdf5'
require 'nn'
require 'optim'
require 'lfs'

local function get_layer(model, name)
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
local function build_model(w2v, opts)
  local ModelBuilder = require 'model.convNN'
  local model_builder = ModelBuilder.new()

  local model
  if opts.warm_start_model == '' then
    model = model_builder:make_net(w2v, opts)
  else
    model = torch.load(opts.warm_start_model).model
  end

  local criterion = nn.ClassNLLCriterion()

  -- move to GPU
  if opts.cudnn == 1 then
    model = model:cuda()
    criterion = criterion:cuda()
  end

  -- get layers
  local layers = {}
  layers['linear'] = get_layer(model, 'nn.Linear')
  layers['w2v'] = get_layer(model, 'nn.LookupTable')
  if opts.skip_kernel > 0 then
    layers['skip_conv'] = get_layer(model, 'skip_conv')
  end
  if opts.model_type == 'multichannel' then
    layers['chan1'] = get_layer(model, 'channel1')
  end

  return model, criterion, layers
end

local function train_loop(data, data_label, train, train_label, dev, dev_label, test, test_label, w2v, opts)
  -- Initialize objects
  local Trainer = require 'trainer'
  local trainer = Trainer.new()

  local optim_method
  if opts.optim_method == 'adadelta' then
    optim_method = optim.adadelta
  elseif opts.optim_method == 'adam' then
    optim_method = optim.adam
  end

  local best_model -- save best model
  local fold_dev_scores = {}
  local fold_test_scores = {}

  for fold = 1, opts.folds do
    local timer = torch.Timer()
    local fold_time = timer:time().real

    print()
    print('==> fold ', fold)

    if opts.has_test == 0 then
      -- make train/test data (90/10 split for train/test)
      local N = data:size(1)
      local i_start = math.floor((fold - 1) * 0.1 * N + 1)
      local i_end = math.floor(fold * 0.1 * N)
      test = data:narrow(1, i_start, i_end - i_start + 1)
      test_label = data_label:narrow(1, i_start, i_end - i_start + 1)
      train = torch.cat(data:narrow(1, 1, i_start), data:narrow(1, i_end, N - i_end + 1), 1)
      train_label = torch.cat(data_label:narrow(1, 1, i_start), data_label:narrow(1, i_end, N - i_end + 1), 1)
    end

    if opts.has_dev == 0 then
      -- Run if we don't have dev (we may or may not have test)
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

    -- build model
    local model, criterion, layers = build_model(w2v, opts)

    -- Call getParameters once
    local params, grads = model:getParameters()

    -- Training loop.
    best_model = model:clone()
    local best_epoch = 1
    local best_err = 0.0

    -- Training.
    if opts.test_only == 0 then
      -- Gradient descent state should persist over epochs
      local state = {}
      for epoch = 1, opts.num_epochs do
        local epoch_time = timer:time().real

        -- Train
        local train_err = trainer:train(train, train_label, model, criterion, optim_method, layers, state, params, grads, opts)
        -- Dev
        local dev_err = trainer:test(dev, dev_label, model, criterion, opts)
        if dev_err > best_err then
          best_model = model:clone()
          best_epoch = epoch
          best_err = dev_err 
        end

        if opts.debug == 1 then
          print()
          print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
          print('\n')
        end

        print('epoch:', epoch, 'train perf:', 100*train_err, '%, val perf ', 100*dev_err, '%')
      end

      print('best dev err:', 100*best_err, '%, epoch ', best_epoch)
      table.insert(fold_dev_scores, best_err)
    end

    -- Testing.
    if opts.train_only == 0 then
      local test_err = trainer:test(test, test_label, best_model, criterion, opts)
      print('test perf ', 100*test_err, '%')
      table.insert(fold_test_scores, test_err)
    end

    if opts.debug == 1 then
      print()
      print('time for one fold: ', (timer:time().real - fold_time * 1000), 'ms')
      print('\n')
    end
  end

  return fold_dev_scores, fold_test_scores, best_model
end

local function load_data(opts)
  local data, data_label
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
  elseif opts.has_test == 1 then
    test = f:read('test'):all()
    test_label = f:read('test_label'):all()
    train = f:read('train'):all()
    train_label = f:read('train_label'):all()
  else
    -- Need CV split
    data = f:read('data'):all()
    data_label = f:read('data_label'):all()
  end
  print('data loaded!')

  return data, data_label, test, test_label, train, train_label, dev, dev_label, w2v
end

local function main()
  -- Flags
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text()
  cmd:text('Convolutional net for sentence classification')
  cmd:text()
  cmd:text('Options')
  cmd:option('-num_epochs', 25, 'Number of training epochs')
  cmd:option('-model_type', 'rand', 'Model type. Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')
  cmd:option('-data', 'MR.hdf5', 'Training data and word2vec data')
  cmd:option('-cudnn', 0, 'Use cudnn and GPUs if set to 1, otherwise set to 0')
  cmd:option('-seed', -1, 'random seed, set -1 for actual random')
  cmd:option('-folds', 10, 'number of folds to use. max 10')
  cmd:option('-debug', 0, 'print debugging info including timing, confusions')
  cmd:option('-gpuid', 1, 'GPU device id to use.')
  cmd:option('-savefile', '', 'Name of output file, which will hold the trained model, model parameters, and training scores. Default filename is TIMESTAMP_results')
  cmd:option('-has_test', 0, 'If data has test, we use it. Otherwise, we use CV on folds')
  cmd:option('-has_dev', 0, 'If data has dev, we use it, otherwise we split from train')
  cmd:option('-zero_indexing', 0, 'If data is zero indexed')
  cmd:text()

  -- Training own dataset
  cmd:option('-train_only', 0, 'Set to 1 to only train model (no testing scores)')
  cmd:option('-test_only', 0, 'Set to 1 to only do testing. Must have a -warm_start_model')
  cmd:option('-warm_start_model', '', 'Path to .t7 file with pre-trained model. Should be loaded with key model')
  cmd:text()

  -- Training options
  cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
  cmd:option('-L2s', 3, 'L2 normalize weights')
  cmd:option('-batch_size', 50, 'Batch size for training')
  cmd:text()

  -- Model options
  cmd:option('-vocab_size', 18766, 'Vocab size')
  cmd:option('-vec_size', 300, 'word2vec vector size')
  cmd:option('-max_sent', 59, 'maximum sentence length')
  cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
  cmd:option('-kernel1', 3, 'Kernel size of convolution 1')
  cmd:option('-kernel2', 4, 'Kernel size of convolution 2')
  cmd:option('-kernel3', 5, 'Kernel size of convolution 3')
  cmd:option('-skip_kernel', 0, 'Use skip kernel')
  cmd:option('-dropout_p', 0.5, 'p for dropout')
  cmd:option('-highway_mlp', 0, 'Number of highway MLP layers')
  cmd:option('-highway_conv_layers', 0, 'Number of highway MLP layers')
  cmd:option('-num_classes', 2, 'Number of output classes')
  cmd:text()

  -- parse arguments
  local opts = cmd:parse(arg)

  if opts.seed ~= -1 then
    torch.manualSeed(opts.seed)
  end
  if opts.cudnn == 1 then
    require 'cutorch'
    if opts.seed ~= -1 then
      cutorch.manualSeedAll(opts.seed)
    end
    cutorch.setDevice(opts.gpuid)
  end

  if opts.test_only == 1 then
    assert(opts.warm_start_model ~= '', 'must have -warm_start_model for testing')
    opts.has_test = 1
  end

  -- Read HDF5 training data
  local data, data_label
  local test, test_label
  local train, train_label
  local dev, dev_label
  local w2v
  data, data_label, test, test_label, train, train_label, dev, dev_label, w2v = load_data(opts)

  opts.vocab_size = w2v:size(1)
  opts.vec_size = w2v:size(2)
  if data ~= nil then
    opts.max_sent = data:size(2)
  else
    opts.max_sent = train:size(2)
  end
  print('vocab size: ', opts.vocab_size)
  print('vec size: ', opts.vec_size)

  if opts.zero_indexing == 1 then
    if data ~= nil then
      data:add(1)
      data_label:add(1)
    else
      train:add(1)
      train_label:add(1)
    end
  end

  if opts.train_only == 1 then
    -- don't do testing if we train_only
    opts.has_test = 1
  end

  if opts.has_test == 1 then
    -- don't do CV if we have a test set
    opts.folds = 1
  end

  -- training loop
  local fold_dev_scores, fold_test_scores, best_model = train_loop(data, data_label, train, train_label, dev, dev_label, test, test_label, w2v, opts)

  if opts.test_only == 0 then
    print('dev scores:')
    print(fold_dev_scores)
    print('average dev score: ', torch.Tensor(fold_dev_scores):mean())
  end

  if opts.train_only == 0 then
    print('test scores:')
    print(fold_test_scores)
    print('average test score: ', torch.Tensor(fold_test_scores):mean())
  end

  -- make sure output directory exists
  if not path.exists('results') then lfs.mkdir('results') end

  local savefile
  if opts.savefile ~= '' then
    savefile = opts.savefile
  else
    savefile = string.format('results/%s_model.t7', os.date('%Y%m%d_%H%M'))
  end
  print('saving results to ', savefile)

  local save = {}
  save['dev_scores'] = fold_dev_scores
  if opts.train_only == 0 then
    save['test_scores'] = fold_test_scores
  end
  save['opts'] = opts
  save['model'] = best_model
  save['embeddings'] = get_layer(best_model, 'nn.LookupTable').weight
  torch.save(savefile, save)
end

main()
