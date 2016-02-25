local HighwayConv = {}

function HighwayConv.conv(vec_size, max_sent, kernel_size, num_layers, bias, f)
  -- size = dimensionality of inputs
  -- num_layers = number of conv layers (default = 1)
  -- bias = bias for transform gate (default = -2)
  -- f = non-linearity (default = ReLU)

  local reshape1, conv, pad, reshape2
  local output, transform_gate, carry_gate
  local num_layers = num_layers or 1
  local bias = bias or -2
  local f = f or cudnn.ReLU()
  local input = nn.Identity()()
  local inputs = {[1]=input}

  for i = 1, num_layers do        
      -- Reshape for spatial convolution
    reshape1 = nn.Reshape(1, max_sent, vec_size, true)
    conv = cudnn.SpatialConvolution(1, vec_size, vec_size, kernel_size)
    pad = nn.Padding(3,kernel_size-1)
    reshape2 = nn.Reshape(max_sent, vec_size, true)
    conv.weight:uniform(-0.01, 0.01)
    conv.bias:zero()

    output = f(conv(reshape1(inputs[i])))
    output = reshape2(pad(output))
    transform_gate = cudnn.Sigmoid()(nn.AddConstant(bias)(conv(reshape1(inputs[i]))))
    transform_gate = reshape2(pad(transform_gate))
    carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))

    output = nn.CAddTable()({
      nn.CMulTable()({transform_gate, output}),
      nn.CMulTable()({carry_gate, inputs[i]})  })
    table.insert(inputs, output)
  end

  return nn.gModule({input},{output})
end

return HighwayConv 
