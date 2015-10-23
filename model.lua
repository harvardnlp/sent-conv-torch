require 'torch'
require 'nn'
require 'nngraph'
local HighwayMLP = require 'highway_mlp'

local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder.init_cmd(cmd)
  cmd:option('-vocab_size', 18766, 'Vocab size')
  cmd:option('-vec_size', 300, 'word2vec vector size')

  cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
  cmd:option('-kernel1', 3, 'Kernel size of convolution 1')
  cmd:option('-kernel2', 4, 'Kernel size of convolution 2')
  cmd:option('-kernel3', 5, 'Kernel size of convolution 3')
  cmd:option('-dropout_p', 0.5, 'p for dropout')
  cmd:option('-highway_layers', 0, 'Number of highway layers')
  cmd:option('-num_classes', 2, 'Number of output classes')
end

function ModelBuilder:make_net(w2v, opts)
  if opts.cudnn == 1 then
    require 'cudnn'
    require 'cunn'
  end

  local input = nn.Identity()()

  local lookup = nn.LookupTable(opts.vocab_size, opts.vec_size)
  if opts.model_type == 'static' or opts.model_type == 'nonstatic' then
    lookup.weight = w2v
  else
    lookup.weight:uniform(-0.25, 0.25)
  end
  -- padding should always be 0
  lookup.weight[1]:zero()

  local lookup_layer = lookup(input)

  -- kernels is an array of kernel sizes
  local kernels = {opts.kernel1, opts.kernel2, opts.kernel3}
  local layer1 = {}
  for i = 1, #kernels do
    local conv
    local conv_layer
    local max_pool
    if opts.cudnn == 1 then
      conv = cudnn.SpatialConvolution(1, opts.num_feat_maps, opts.vec_size, kernels[i])

      -- Reshape for spatial convolution
      conv_layer = nn.Reshape(opts.num_feat_maps, -1, true)(conv(nn.Reshape(1, -1, opts.vec_size, true)(lookup_layer)))
      max_pool = cudnn.ReLU()(nn.Max(3)(conv_layer))
    else
      conv = nn.TemporalConvolution(opts.vec_size, opts.num_feat_maps, kernels[i])
      conv_layer = conv(lookup_layer)
      --model:add(nn.Transpose({2,3})) -- swap feature maps and time
      max_pool = nn.Max(2)(nn.ReLU()(conv)) -- max over time
    end

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    table.insert(layer1, max_pool)
  end

  local conv_layer_concat
  if #kernels > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
  else
    conv_layer_concat = layer1[1]
  end

  local softmax
  if opts.cudnn == 1 then
    softmax = cudnn.LogSoftMax()
  else
    softmax = nn.LogSoftMax()
  end

  local last_layer = conv_layer_concat
  if opts.highway_layers > 0 then
    -- use highway layers
    local highway = HighwayMLP.mlp((#kernels) * opts.num_feat_maps, opts.highway_layers)
    last_layer = highway(conv_layer_concat)
  end

  -- simple MLP layer
  local linear = nn.Linear((#kernels) * opts.num_feat_maps, opts.num_classes)
  linear.weight:uniform(-0.01, 0.01)
  linear.bias:zero()
  local output = softmax(linear(nn.Dropout(opts.dropout_p)(last_layer)))

  self.model = nn.gModule({input}, {output})
  return self.model
end

function ModelBuilder:get_linear()
  if not self.model then return end

  local linear
  function get_layer(layer)
    if torch.typename(layer) == 'nn.Linear' then
      linear = layer
    end
  end

  self.model:apply(get_layer)
  return linear
end

function ModelBuilder:get_w2v()
  if not self.model then return end

  local w2v
  function get_layer(layer)
    if torch.typename(layer) == 'nn.LookupTable' then
      w2v = layer
    end
  end

  self.model:apply(get_layer)
  return w2v
end

return ModelBuilder
