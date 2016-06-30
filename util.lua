function get_layer(model, name)
  local named_layer
  function get(layer)
    if layer.name == name or torch.typename(layer) == name then
      named_layer = layer
    end
  end

  model:apply(get)
  return named_layer
end

