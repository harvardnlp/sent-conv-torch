require 'nn'
require 'optim'
require 'sys'
require 'cutorch'

local Trainer = torch.class('Trainer')

function Trainer:__init()
  self.optim_method = optim.adadelta
  self.config = {
    eps = 1e-10
  }
  self.state = {}
  self.L2s = 3

  self.batch_size = 50
  model_type = 'static'
end

-- Perform one epoch of training.
function Trainer:train(train_data, train_labels, model, criterion)
  model:training()

  params, grads = model:getParameters()

  local train_size = train_data:size(1)

  local time = sys.clock()
  local total_err = 0

  for t = 1, train_size, self.batch_size do
    --print('Batch ' .. t)
    -- data samples and labels, in mini batches.
    local batch_size = math.min(self.batch_size, train_size - t + 1)
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
    self.optim_method(func, params, self.config, self.state)

    -- Renorm (Euclidean projection to L2 ball)
    local w = model.modules[7].weight -- TODO(jeffreyling): bad constant
    local n = w:view(w:size(1)*w:size(2)):norm()
    if (n > self.L2s) then 
      w:div(self.L2s * n)
    end
  end

  print(confusion)
  print('Total err: ' .. total_err / train_size)

  -- time taken
  time = sys.clock() - time
  time = time / train_size
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
end

function Trainer:test(test_data, test_labels, model, criterion)
  model:evaluate()

  local classes = {'1', '2'}
  local confusion = optim.ConfusionMatrix(classes)
  confusion:zero()

  local test_size = test_data:size(1)

  local total_err = 0

  for t = 1, test_size, self.batch_size do
    -- data samples and labels, in mini batches.
    local batch_size = math.min(self.batch_size, test_size - t + 1)
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
