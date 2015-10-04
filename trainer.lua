require 'nn'
require 'optim'
require 'sys'

local trainer = torch.class('trainer')

function trainer:__init()
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
function trainer:train(train_data, model, criterion)
  print('==> training epoch...')
  model:training()

  local classes = {'1', '2'}
  local confusion = optim.ConfusionMatrix(classes)
  local params, grads = model:getParameters()

  local train_size = train_data.data:size(1)
  local shuffle = torch.randperm(train_size)

  local time = sys.clock()

  for t = 1, train_size, self.batch_size do
    print('Batch ' .. t)
    -- data samples and labels, in mini batches.
    local inputs = {}
    local targets = {}
    for i = t, math.min(t + self.batch_size - 1, train_size) do
      local input = train_data.data[shuffle[i]]
      local target = train_data.labels[shuffle[i]]
      -- TODO(jeffreyling): use cuda
      input = input:double()
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    -- closure to return f, df/dx
    local func = function(x)
      -- get new parameters
      if x ~= params then
        params:copy(x)
      end
      -- reset gradients
      grads:zero()

      -- average of all criterions
      local f = 0

      for i = 1,#inputs do
        -- estimate f
        local output = model:forward(inputs[i])
        local err = criterion:forward(output, targets[i])
        f = f + err

        -- estimate df/dw
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)

        -- update confusion
        confusion:add(output, targets[i])
      end

      -- normalize gradients
      grads:div(#inputs)
      f = f/#inputs

      return f, grads
    end

    -- gradient descent
    self.optim_method(func, params, self.config, self.state)

    -- Renorm (Euclidean projection to L2 ball)
    local w = model.modules[8].weight -- TODO(jeffreyling): bad constant
    local n = w:view(w:size(1)*w:size(2)):norm()
    if (n > self.L2s) then 
      w:div(self.L2s * n)
    end

    print(confusion)
  end

  -- time taken
  time = sys.clock() - time
  time = time / train_size
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
end

function trainer:test(test_data)
  print('==> testing...')
  model:evaluate()

  --for t = 1, self.test_data:size() do
    ---- get sample
    --local input = test_data.data[t]
    --input = input:double()
  --end
end

return trainer
