require 'optim'
require 'sys'

local trainer = torch.class('trainer')

function trainer:__init()
  self.model = require 'model'
  self.train_data, self.test_data = require 'data'

  self.criterion = nn.ClassNLLCriterion()
  self.optim_method = optim.adadelta
  -- TODO(jeffreyling): Figure out what goes in state.
  self.training_state = {
    learningRate = 1e-3,
    paramVariance = 0.5
  }
  self.L2s = 3

  self.batch_size = 50
  self.model_type = 'static'
end

function trainer:train()
  print('==> training net...')
  self.model:training()

  local time = sys.clock()
  params, grads = self.model:getParameters()

  shuffle = torch.randperm(self.train_data:size(1))

  -- do one epoch
  epoch = epoch or 1
  print('==> epoch #' .. epoch .. ' on training data')

  for t = 1, self.train_data:size(1), self.batch_size do
    -- data samples and labels, in mini batches.
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+self.batch_size-1, train_size) do
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
        local output = self.model:forward(inputs[i])
        local err = self.criterion:forward(output, targets[i])
        f = f + err

        -- estimate df/dw
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)
      end

      -- normalize gradients
      grads:div(#inputs)
      f = f/#inputs

      return f, grads
    end

    -- gradient descent
    self.optim_method(func, params, self.training_state)

    -- Renorm (Euclidean projection to L2 ball)
    local n = params:view(params:size(1)*params:size(2)):norm()
    if (n > self.L2s) then 
      params:div(self.L2s * n)
    end
  end

  -- time taken
  time = sys.clock() - time
  time = time / train_data:size(1)
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')
end

function trainer:evaluate()
  print('==> testing net on sample...')
  self.model:evaluate()
  self.model:forward(sentence)
end

return trainer
