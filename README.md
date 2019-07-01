# QM
SwapHead should have intNet

untie weights
m = F.relu(self.conv1(out, data.edge_index, edge_attr))
m = F.relu(self.linear(m))
out = out + m
