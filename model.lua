require 'torch'
require 'nn'

local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder.init_cmd(cmd)
  cmd:option('-vocab_size', 21421, 'Vocab size')
  cmd:option('-vec_size', 300, 'word2vec vector size')

  cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
  cmd:option('-kernel_size', 3, 'Kernel size of convolution')
  cmd:option('-dropout_p', 0.5, 'p for dropout')
  cmd:option('-num_classes', 2, 'Number of output classes')
end

function ModelBuilder:make_net(w2v, opts)
  self.model = nn.Sequential()
  local model = self.model

  local lookup = nn.LookupTable(opts.vocab_size, opts.vec_size)
  if opts.model_type == 'static' or opts.model == 'nonstatic' then
    lookup.weights = w2v
  end
  model:add(lookup)
    
  if opts.cudnn == 1 then
    require 'cudnn'
    -- Reshape for spatial convolution
    model:add(nn.Reshape(1, -1, opts.vec_size, true))
    model:add(cudnn.SpatialConvolution(1, opts.num_feat_maps, opts.vec_size, opts.kernel_size))
    model:add(nn.Reshape(opts.num_feat_maps, -1, true))
    model:add(nn.Max(3))
    model:add(cudnn.ReLU())
  else
    model:add(nn.TemporalConvolution(opts.vec_size, opts.num_feat_maps, opts.kernel_size))
    --model:add(nn.TemporalConvolutionFB(opts.vec_size, opts.num_feat_maps, opts.kernel_size))
    model:add(nn.ReLU())
    model:add(nn.Transpose({2,3})) -- swap feature maps and time
    model:add(nn.Max(3)) -- max over time
  end

  model:add(nn.Dropout(opts.dropout_p))
  model:add(nn.Linear(opts.num_feat_maps, opts.num_classes))
  
  if opts.cudnn == 1 then
    model:add(cudnn.LogSoftMax())
  else
    model:add(nn.LogSoftMax())
  end

  return self.model
end

function ModelBuilder:get_linear()
  if not self.model then return end

  for i = 1, #self.model do
    if torch.typename(self.model.modules[i]) == 'nn.Linear' then
      return self.model.modules[i]
    end
  end
end

function ModelBuilder:get_w2v()
  if not self.model then return end

  for i = 1, #self.model do
    if torch.typename(self.model.modules[i]) == 'nn.LookupTable' then
      return self.model.modules[i]
    end
  end
end

return ModelBuilder
