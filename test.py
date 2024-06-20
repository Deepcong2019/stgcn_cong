import pickle

import numpy as np
import torch
import torch.nn as nn
from net.utils.graph import Graph
from net.utils.tgcn import ConvTemporalGraphical
# from net.st_gcn import Model
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self, in_channels=3, num_class=6,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(layout='my_pose', strategy='spatial',)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # 寄存器变量访问快
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x
    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A

if __name__ == '__main__':
    # set params
    weights_path = 'work_dir/recognition/kinetics_skeleton/ST_GCN/epoch140_model.pt'
    # label_list = ['standing', 'walking', 'laying']
    label_list = []
    with open('resource/kinetics_skeleton/label_name_KTH.txt', 'r') as f:
        for line in f:
            label_list.append(line.strip('\n'))
    data_path = "data/KTH/kinetics-skeleton/val_data.npy"
    label_path = "data/KTH/kinetics-skeleton/val_label.pkl"

    model = Model().to('cuda:0')
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model.eval()

    val_data = np.load(data_path)
    f = open(label_path, 'rb')
    label_data = pickle.load(f)
    Num_ture = [0] * (len(label_list)+1)
    Num_total = [0] * (len(label_list)+1)
    '''
    data1 = torch.tensor(val_data[0]).unsqueeze(0)
    data1 = data1.float().to("cuda:0")
    traced_model = torch.jit.trace(model, data1)  #trace模型
    '''
    for data in val_data:
        data = torch.tensor(data).unsqueeze(0)
        data = data.float().to("cuda:0")
        output = model(data).data.cpu().numpy()[0].tolist()
        pred_index = output.index(max(output))
        label_index = label_data[1][Num_total[-1]]
        print("Label/Pred:  {}/{}".format(label_list[label_index], label_list[pred_index]))

        for l_idx in range(len(label_list)):
            if label_index == l_idx:
                Num_total[l_idx] += 1
                if pred_index==label_index:
                    Num_ture[l_idx] += 1
                    Num_ture[-1] += 1
        Num_total[-1] += 1

    for idx in range(len(label_list)):
        print("Accuracy for {}-{}: {}, TP: {}, Total_num: {}".format(idx, label_list[idx], Num_ture[idx]/Num_total[idx], Num_ture[idx], Num_total[idx]))
    print("Total Accuracy: {}".format(Num_ture[-1]/Num_total[-1]))
    # # traced_model.save("stgcn_torchscript.pt")  #保存trace的模型
