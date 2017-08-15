require 'torch'
require 'nn'
require 'nngraph'

function make_net(w2v, opt)
  if opt.cudnn == 1 then
    require 'cudnn'
    require 'cunn'
  end

  local input = nn.Identity()()

  local lookup
  if opt.model_type == 'multichannel' then
    local channels = {}
    for i = 1, 2 do
      local chan = nn.LookupTable(opt.vocab_size, opt.vec_size)
      chan.weight:copy(w2v)
      chan.weight[1]:zero()
      chan.name = 'channel' .. i
      table.insert(channels, chan(input))
    end
    lookup = channels
  else
    lookup = nn.LookupTable(opt.vocab_size, opt.vec_size)
    if opt.model_type == 'static' or opt.model_type == 'nonstatic' then
      lookup.weight:copy(w2v)
    else
      -- rand
      lookup.weight:uniform(-0.25, 0.25)
    end
    -- padding should always be 0
    lookup.weight[1]:zero()

    lookup = lookup(input)
  end

  -- kernels is an array of kernel sizes
  local kernels = opt.kernels
  local layer1 = {}
  for i = 1, #kernels do
    local conv
    local conv_layer
    local max_time
    if opt.cudnn == 1 then
      conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kernels[i])
      if opt.model_type == 'multichannel' then
        local lookup_conv = {}
        for chan = 1, 2 do
          table.insert(lookup_conv, nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
            conv(
            nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
            lookup[chan]))))
        end
        conv_layer = nn.CAddTable()(lookup_conv)
        max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
      else
        if opt.highway_conv_layers > 0 then
          -- Highway conv layers
          local HighwayConv = require 'model/highway_conv'
          local highway_conv = HighwayConv.conv(opt.vec_size, opt.max_sent, kernels[i], opt.highway_conv_layers)
          conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
            conv(nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
            highway_conv(lookup))))
          max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
        else
          conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kernels[i]+1, true)(
            conv(
            nn.Reshape(1, opt.max_sent, opt.vec_size, true)(
            lookup)))
          max_time = nn.Max(3)(cudnn.ReLU()(conv_layer))
        end
      end
    else
      conv = nn.TemporalConvolution(opt.vec_size, opt.num_feat_maps, kernels[i])
      if opt.model_type == 'multichannel' then
        local lookup_conv = {}
        for chan = 1,2 do
          table.insert(lookup_conv, conv(lookup[chan]))
        end
        conv_layer = nn.CAddTable()(lookup_conv)
        max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
      else
        conv = nn.TemporalConvolution(opt.vec_size, opt.num_feat_maps, kernels[i])
        conv_layer = conv(lookup)
        max_time = nn.Max(2)(nn.ReLU()(conv_layer)) -- max over time
      end
    end

    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()
    conv.name = 'convolution'
    table.insert(layer1, max_time)
  end

  if opt.skip_kernel > 0 then
    -- skip kernel
    local kern_size = 5 -- fix for now
    local skip_conv = cudnn.SpatialConvolution(1, opt.num_feat_maps, opt.vec_size, kern_size)
    skip_conv.name = 'skip_conv'
    skip_conv.weight:uniform(-0.01, 0.01)
    -- skip center for now
    skip_conv.weight:select(3,3):zero()
    skip_conv.bias:zero()
    local skip_conv_layer = nn.Reshape(opt.num_feat_maps, opt.max_sent-kern_size+1, true)(skip_conv(nn.Reshape(1, opt.max_sent, opt.vec_size, true)(lookup)))
    table.insert(layer1, nn.Max(3)(cudnn.ReLU()(skip_conv_layer)))
  end

  local conv_layer_concat
  if #layer1 > 1 then
    conv_layer_concat = nn.JoinTable(2)(layer1)
  else
    conv_layer_concat = layer1[1]
  end

  local last_layer = conv_layer_concat
  if opt.highway_mlp > 0 then
    -- use highway layers
    local HighwayMLP = require 'model/highway_mlp'
    local highway = HighwayMLP.mlp((#layer1) * opt.num_feat_maps, opt.highway_layers)
    last_layer = highway(conv_layer_concat)
  end

  -- simple MLP layer
  local linear = nn.Linear((#layer1) * opt.num_feat_maps, opt.num_classes)
  linear.weight:normal():mul(0.01)
  linear.bias:zero()

  local output = nn.LogSoftMax()(linear(nn.Dropout(opt.dropout_p)(last_layer))) 
  model = nn.gModule({input}, {output})
  return model
end
