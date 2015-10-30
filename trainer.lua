require 'nn'
require 'sys'
require 'torch'

local Trainer = torch.class('Trainer')

function Trainer.init_cmd(cmd)
  cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta, adam')
  cmd:option('-L2s', 3, 'L2 normalize weights')
  cmd:option('-batch_size', 50, 'Batch size for training')
end

-- Perform one epoch of training.
function Trainer:train(train_data, train_labels, model, criterion, optim_method, layers, opts)
  model:training()

  params, grads = model:getParameters()
  _, w2v_grads = layers.w2v:getParameters()

  local train_size = train_data:size(1)

  local time = sys.clock()
  local total_err = 0

  local classes = {}
  for i = 1, opts.num_classes do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local config -- for optim
  if opts.optim_method == 'adadelta' then
    config = { rho = 0.95, eps = 1e-6 } 
  elseif opts.optim_method == 'adam' then
    config = {}
  end

  -- shuffle batches
  local num_batches = math.floor(train_size / opts.batch_size)
  local shuffle = torch.randperm(num_batches)
  for i = 1, shuffle:size(1) do
    local t = (shuffle[i] - 1) * opts.batch_size + 1
    local batch_size = math.min(opts.batch_size, train_size - t + 1)

    -- data samples and labels, in mini batches.
    local inputs = train_data:narrow(1, t, batch_size)
    local targets = train_labels:narrow(1, t, batch_size)
    if opts.cudnn == 1 then
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
      for i = 1, batch_size do
        confusion:add(outputs[i], targets[i])
      end

      -- compute gradients
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      if opts.model_type == 'static' then
        -- don't update embeddings for static model
        w2v_grads:zero()
      end

      return err, grads
    end

    -- gradient descent
    optim_method(func, params, config)
    -- reset padding embedding to zero
    layers.w2v.weight[1]:zero()
    -- keep skip kernel at zero
    layers.skip_conv.weight:select(3,3):zero()

    -- Renorm (Euclidean projection to L2 ball)
    local renorm = function(row)
      local n = row:norm()
      if (n > opts.L2s) then
        row:mul(opts.L2s):div(1e-7 + n)
      end
    end

    -- renormalize linear row weights
    local w = layers.linear.weight
    for i = 1, w:size(1) do
      renorm(w[i])
    end
  end

  if opts.debug == 1 then
    print('Total err: ' .. total_err / train_size)
    print(confusion)
  end

  -- time taken
  time = sys.clock() - time
  time = opts.batch_size * time / train_size
  if opts.debug == 1 then
    print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end

function Trainer:test(test_data, test_labels, model, criterion, opts)
  model:evaluate()

  local classes = {}
  for i = 1, opts.num_classes do
    table.insert(classes, i)
  end
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local test_size = test_data:size(1)

  local total_err = 0

  for t = 1, test_size, opts.batch_size do
    -- data samples and labels, in mini batches.
    local batch_size = math.min(opts.batch_size, test_size - t + 1)
    local inputs = test_data:narrow(1, t, batch_size)
    local targets = test_labels:narrow(1, t, batch_size)
    if opts.cudnn == 1 then
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

  if opts.debug == 1 then
    print(confusion)
    print('Total err: ' .. total_err / test_size)
  end

  -- return error percent
  confusion:updateValids()
  return confusion.totalValid
end

function Trainer.adadelta(opfunc, x, config, state)
       --(0) get/update state
  if config == nil and state == nil then
    print('no state table, ADADELTA initializing')
  end

  local config = config or {}
  local state = state or config
  local rho = config.rho or 0.9
  local eps = config.eps or 1e-6
  state.evalCounter = state.evalCounter or 0
  -- (1) evaluate f(x) and df/dx
  local fx,dfdx = opfunc(x)

  -- (2) parameter update
  if not state.paramVariance then
  state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  state.delta = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  state.accDelta = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  end
  state.paramVariance:mul(rho):addcmul(1-rho,dfdx,dfdx)
  state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):add(eps):sqrt()
  state.delta:resizeAs(state.paramVariance):copy(state.accDelta):add(eps):sqrt():cdiv(state.paramStd + eps):cmul(dfdx)
  x:add(-1, state.delta)
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)
  -- (3) update evaluation counter
  state.evalCounter = state.evalCounter + 1

  -- return x*, f(x) before optimization
  return x,{fx}
end

return Trainer
