-- Process training and testing data.
-- Returns them in train_data and test_data.

require 'hdf5'

local f = hdf5.open('rt-polarity.hdf5', 'r')
local train = f:read('train'):all()
local train_label = f:read('train_label'):all()
local word_idx_map = f:read('word_idx_map'):all()

-- Format is
-- *  'data': 2-d tensor, train size by max sentence len
-- *  'label': 1-d tensor, train size
local train_data = {
  data = train,
  labels = train_label
}
local test_data = {
  data = test,
  labels = test_label
}

return train_data, test_data
