require 'nn'
require 'sys'
require 'torch'

local Trainer = torch.class('Trainer')

-- Perform one epoch of training.
function Trainer:train(train_data, train_labels, model, criterion, optim_method, layers, state, params, grads)
  model:training()

  local train_size = train_data:size(1)

  local timer = torch.Timer()
  local time = timer:time().real
  local total_err = 0

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local config -- for optim
  if opt.optim_method == 'adadelta' then
    config = { rho = 0.95, eps = 1e-6 } 
  elseif opt.optim_method == 'adam' then
    config = {}
  end

  -- shuffle batches
  local num_batches = math.floor(train_size / opt.batch_size)
  local shuffle = torch.randperm(num_batches)
  for i = 1, shuffle:size(1) do
    local t = (shuffle[i] - 1) * opt.batch_size + 1
    local batch_size = math.min(opt.batch_size, train_size - t + 1)

    -- data samples and labels, in mini batches.
    local inputs = train_data:narrow(1, t, batch_size)
    local targets = train_labels:narrow(1, t, batch_size)
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
      targets = targets:cuda()
    else
      inputs = inputs:double()
      targets = targets:double()
    end

    -- closure to return err, df/dx
    local func = function(x)
      -- get new parameters
      if x ~= params then
        params:copy(x)
      end
      -- reset gradients
      grads:zero()

      -- forward pass
      local outputs = model:forward(inputs)
      local err = criterion:forward(outputs, targets)

      -- track errors and confusion
      total_err = total_err + err * batch_size
      for j = 1, batch_size do
        confusion:add(outputs[j], targets[j])
      end

      -- compute gradients
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      if opt.model_type == 'static' then
        -- don't update embeddings for static model
        layers.w2v.gradWeight:zero()
      elseif opt.model_type == 'multichannel' then
        -- keep one embedding channel static
        layers.chan1.gradWeight:zero()
      end

      return err, grads
    end

    -- gradient descent
    optim_method(func, params, config, state)
    -- reset padding embedding to zero
    layers.w2v.weight[1]:zero()
    if opt.skip_kernel > 0 then
      -- keep skip kernel at zero
      layers.skip_conv.weight:select(3,3):zero()
    end

    -- Renorm (Euclidean projection to L2 ball)
    local renorm = function(row)
      local n = row:norm()
      row:mul(opt.L2s):div(1e-7 + n)
    end

    -- renormalize linear row weights
    local w = layers.linear.weight
    for j = 1, w:size(1) do
      renorm(w[j])
    end
  end

  if opt.debug == 1 then
    print('Total err: ' .. total_err / train_size)
    print(confusion)
  end

  -- time taken
  time = timer:time().real - time
  time = opt.batch_size * time / train_size
  if opt.debug == 1 then
    print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end

function Trainer:test(test_data, test_labels, model, criterion)
  model:evaluate()

  local classes = {}
  for i = 1, opt.num_classes do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local test_size = test_data:size(1)

  local total_err = 0

  for t = 1, test_size, opt.batch_size do
    -- data samples and labels, in mini batches.
    local batch_size = math.min(opt.batch_size, test_size - t + 1)
    local inputs = test_data:narrow(1, t, batch_size)
    local targets = test_labels:narrow(1, t, batch_size)
    if opt.cudnn == 1 then
      inputs = inputs:cuda()
      targets = targets:cuda()
    else
      inputs = inputs:double()
      targets = targets:double()
    end

    local outputs = model:forward(inputs)
    local err = criterion:forward(outputs, targets)
    total_err = total_err + err * batch_size

    for i = 1, batch_size do
      confusion:add(outputs[i], targets[i])
    end
  end

  if opt.debug == 1 then
    print(confusion)
    print('Total err: ' .. total_err / test_size)
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end

return Trainer
