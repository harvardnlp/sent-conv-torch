require 'nn'
require 'sys'
require 'torch'

local Trainer = torch.class('Trainer')

function Trainer.init_cmd(cmd)
  cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta')
  cmd:option('-L2s', 3, 'L2 normalize weights')
  cmd:option('-batch_size', 32, 'Batch size for training')
end

-- Perform one epoch of training.
function Trainer:train(train_data, train_labels, model, criterion, optim_method, layers, opts)
  model:training()

  params, grads = model:getParameters()
  _, w2v_grads = layers.w2v:getParameters()

  local train_size = train_data:size(1)

  local time = sys.clock()
  local total_err = 0

  local config = {} -- for optim
  for t = 1, train_size, opts.batch_size do
    --print('Batch ' .. t)
    -- data samples and labels, in mini batches.
    local batch_size = math.min(opts.batch_size, train_size - t + 1)
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

      -- compute gradients
      local outputs = model:forward(inputs)
      local err = criterion:forward(outputs, targets)

      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      if opts.model_type == 'static' then
        -- don't update embeddings for static model
        w2v_grads:zero()
      end

      total_err = total_err + err * batch_size
      return err, grads
    end

    -- gradient descent
    optim_method(func, params, config)

    -- Renorm (Euclidean projection to L2 ball)
    local w = layers.linear.weight
    local n = w:view(w:size(1)*w:size(2)):norm()
    if (n > opts.L2s) then 
      w:mul(opts.L2s):div(n)
    end
  end

  print('Total err: ' .. total_err / train_size)

  -- time taken
  time = sys.clock() - time
  time = opts.batch_size * time / train_size
  print("==> time to learn 1 batch = " .. (time*1000) .. 'ms')
end

function Trainer:test(test_data, test_labels, model, criterion, opts)
  model:evaluate()

  local classes = {'1', '2'}
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

  print(confusion)
  print('Total err: ' .. total_err / test_size)
end

return Trainer
