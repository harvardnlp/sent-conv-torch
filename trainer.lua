require 'nn'
require 'sys'
require 'cutorch'
require 'torch'

local Trainer = torch.class('Trainer')

function Trainer.init_cmd(cmd)
  cmd:option('-optim_method', 'adadelta', 'Gradient descent method. Options: adadelta')
  cmd:option('-L2s', 3, 'L2 normalize weights')
  cmd:option('-batch_size', 50, 'Batch size for training')
end

-- Perform one epoch of training.
function Trainer:train(train_data, train_labels, model, criterion, optim_method, linear, opts)
  model:training()

  params, grads = model:getParameters()

  local train_size = train_data:size(1)

  local time = sys.clock()
  local total_err = 0

  for t = 1, train_size, opts.batch_size do
    --print('Batch ' .. t)
    -- data samples and labels, in mini batches.
    local batch_size = math.min(opts.batch_size, train_size - t + 1)
    local inputs = train_data:narrow(1, t, batch_size)
    local targets = train_labels:narrow(1, t, batch_size)
    inputs = inputs:cuda()
    targets = targets:cuda()

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

      total_err = total_err + err * batch_size
      return err, grads
    end

    -- gradient descent
    optim_method(func, params, {}, {})

    -- Renorm (Euclidean projection to L2 ball)
    local w = linear.weight -- TODO(jeffreyling): bad constant
    local n = w:view(w:size(1)*w:size(2)):norm()
    if (n > opts.L2s) then 
      w:mul(opts.L2s):div(n)
    end
  end

  print('Total err: ' .. total_err / train_size)

  -- time taken
  time = sys.clock() - time
  time = self.batch_size * time / train_size
  print("\n==> time to learn 1 batch = " .. (time*1000) .. 'ms')
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
    inputs = inputs:cuda()
    targets = targets:cuda()

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
