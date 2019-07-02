# QM
SwapHead should have intNet

untie weights
m = F.relu(self.conv1(bn(out), data.edge_index, edge_attr))
m = F.relu(self.linear(m))
out = out + m

temp_ = torch.cat([data.edge_attr3,data.edge_attr4],1)
edge_attr3 = torch.cat([temp_,temp_],0)
nn2 = Linear(***, dim * dim * 2 * 2, bias=False)
