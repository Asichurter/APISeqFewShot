import torch as t
import torch.nn as nn

class BiLstmCellEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bidirectional=True,
                 self_att_dim=64):

        super(BiLstmCellEncoder, self).__init__()

        self.Bidirectional = bidirectional
        self.NumOfLayers = num_layers

        self.ForwardCells = nn.ModuleList([nn.LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)
                      if i==0 else
                      nn.LSTMCell(input_size=hidden_size,
                                  hidden_size=hidden_size)
                      for i in range(num_layers)])

        if bidirectional:
            self.BackwardCells = nn.ModuleList([nn.LSTMCell(input_size=input_size,
                                  hidden_size=hidden_size)
                      if i==0 else
                      nn.LSTMCell(input_size=hidden_size,
                                  hidden_size=hidden_size)
                      for i in range(num_layers)])
        else:
            self.BackwardCells = None

        self.SelfAttention = SelfAttention(input_size=2*hidden_size if bidirectional else hidden_size,
                                           hidden_size=self_att_dim,
                                           pack=False)

    def forward(self, x):
        # input shape: [batch, seq, dim]
        num_directions = 2 if self.Bidirectional else 1
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_dim = self.ForwardCells[0].hidden_size

        hidden_states = t.empty((self.NumOfLayers, num_directions, batch_size, seq_len, hidden_dim)).cuda()

        for layer in range(self.NumOfLayers):

            # 定义初始状态为0向量
            f_h_x, f_c_x = t.zeros((batch_size, hidden_dim)).cuda(), t.zeros((batch_size, hidden_dim)).cuda()
            if num_directions > 1:
                b_h_x, b_c_x = t.zeros((batch_size, hidden_dim)).cuda(), t.zeros((batch_size, hidden_dim)).cuda()

            # 如果是第一层，则输入就是原始输入
            # 如果是后续层，则输入是上一层的隐藏层
            # hidden shape: [direction, batch, seq_len, ]
            forward_layer_input = x if layer==0 else hidden_states[layer-1][0]
            backward_layer_input = x if layer==0 else hidden_states[layer-1][1]

            # # 每一层都要清空隐藏态存储区
            # hidden_states = t.empty((num_directions, batch_size, seq_len, hidden_dim)).cuda()

            for i in range(seq_len):
                f_h_x, f_c_x = self.ForwardCells[layer](forward_layer_input[:,i,:], (f_h_x, f_c_x))
                hidden_states[layer,0,:,i,:] = f_h_x

                if num_directions > 1:
                    b_h_x, b_c_x = self.BackwardCells[layer](backward_layer_input[:,seq_len-1-i,:], (b_h_x, b_c_x))
                    # 反向的序列需要将隐藏层放置在首位使得与正向隐藏态对齐
                    hidden_states[layer,1,:,seq_len-1-i,:] = b_h_x

            # 隐藏态shape: [direction, seq_len, batch, dim] -> [direction, batch, seq_len, dim]
            # hidden_states = t.cat((hidden_states[0].unsqueeze(0), hidden_states[1].unsqueeze(0)), dim=0).cuda()
            #t.Tensor(hidden_states).cuda()
            # 重整形状以达到batch_first目的
            # hidden_states = hidden_states.transpose(1,2).contiguous()

        # 结束所有层后，将最后一层的两个方向的隐藏态连接起来作为LSTM输出
        # shape: [layer, direction, batch, seq_len, hidden]->[batch, seq_len, hidden*directions]
        if num_directions > 1:
            out = t.cat((hidden_states[-1][0], hidden_states[-1][1]), dim=2)
        else:
            out = hidden_states[-1][0]
        out = out.view(batch_size, seq_len, num_directions*hidden_dim)

        return self.SelfAttention(out)