require 'torch'
require 'nn'
require 'fbcunn'
require 'cunn'

local ModelBuilder = torch.class('ModelBuilder')

function ModelBuilder.init_cmd(cmd)
  cmd:option('-vocab_size', 21421, 'Vocab size')
  cmd:option('-vec_size', 300, 'word2vec vector size')

  cmd:option('-num_feat_maps', 100, 'Number of feature maps after 1st convolution')
  cmd:option('-kernel_size', 3, 'Kernel size of convolution')
  cmd:option('-dropout_p', 0.5, 'p for dropout')
  cmd:option('-num_classes', 2, 'Number of output classes')
end

function ModelBuilder:make_net(opts)
  local model = nn.Sequential()
  model:add(nn.LookupTable(opts.vocab_size, opts.vec_size)) -- LookupTable for word2vec
  --model:add(nn.TemporalConvolution(opts.vec_size, opts.num_feat_maps, opts.kernel_size))
  model:add(nn.TemporalConvolutionFB(opts.vec_size, opts.num_feat_maps, opts.kernel_size))
  model:add(nn.ReLU())
  model:add(nn.Transpose({2,3})) -- swap feature maps and time
  model:add(nn.Max(3)) -- max over time

  model:add(nn.Dropout(opts.dropout_p))
  model:add(nn.Linear(opts.num_feat_maps, opts.num_classes))
  model:add(nn.LogSoftMax())

  return model
end

return ModelBuilder
