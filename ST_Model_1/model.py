import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1) #y = x_1^T A x_2 + b

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x) #正对之间的概率分数
        sc_2 = self.f_k(h_mi, c_x) #负对之间的概率分数

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb) #邻居的基因表达之和
        row_sum = torch.sum(mask, 1) #计算度
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum #除以度
          
        return F.normalize(global_emb, p=2, dim=1)

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
    
class Encoder(Module):
    def __init__(self, nnodes, in_features, nhid1, nhid2, out_features, n_clusters, graph_neigh, dropout=0.1, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.n_clusters = n_clusters
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nnodes = nnodes
        
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.lamda1 = Parameter(torch.FloatTensor(self.nnodes, 1))
        self.lamda2 = Parameter(torch.FloatTensor(self.nnodes, 1))
        self.lamda3 = Parameter(torch.FloatTensor(self.nnodes, 1))
        self.lamda4 = Parameter(torch.FloatTensor(self.nnodes, 1))
        self.weight2_1 = Parameter(torch.FloatTensor(self.in_features, 512))
        self.weight2_2 = Parameter(torch.FloatTensor(512, self.out_features))
        self.weight3_2 = Parameter(torch.FloatTensor(512, 512))

        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        self.attention = Attention(self.out_features)


        self.cluster_projector = nn.Sequential(  #####聚类投影，z_dim维投影到n_clusters维
            nn.Linear(self.out_features, self.n_clusters))  ####32-10
            # nn.Softmax(dim=1))

        self.register_buffer('teacher_centers', torch.zeros(1, self.out_features))
        self.register_buffer('previous_centers', torch.zeros(1, self.out_features))

        self.student_temp = 0.9
        self.teacher_temp = 0.06

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.lamda1)
        torch.nn.init.xavier_uniform_(self.lamda2)
        torch.nn.init.xavier_uniform_(self.lamda3)
        torch.nn.init.xavier_uniform_(self.lamda4)
        torch.nn.init.xavier_uniform_(self.weight2_1)
        torch.nn.init.xavier_uniform_(self.weight2_2)
        torch.nn.init.xavier_uniform_(self.weight3_2)

    def compute_loss(self, teacher_logits, student_logits, eps=1e-20):
        teacher_logits = teacher_logits.detach()
        student_probs = (student_logits / self.student_temp).softmax(dim=-1)
        teacher_probs = ((teacher_logits - self.teacher_centers) / self.teacher_temp).softmax(dim=-1)
        loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim=-1).mean()
        return loss

    def forward(self, feat, feat_a, feat_mask, adj, identify_matrix):
        # 1-hop(1GCN)
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight2_1)
        z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z) # I+A

        z = self.act(z)
        z = F.dropout(z, self.dropout, self.training)
        z = torch.mm(z, self.weight2_2)
        z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z) # I+A


        # 解码基因表达
        hiden_emb = z #原始数据经过GCN得到的潜在表达hiden_emb
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h) #解码后的表达（原始数据得到的）h

        emb = self.act(z) #对中间得到的潜在表达使用激活函数

        # 1-hop(1GCN)
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight2_1)
        z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a) # I+A

        z_a = self.act(z_a)
        z_a = F.dropout(z_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight2_2)
        z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a)  # I+A


        emb_a = self.act(z_a)

        # 1-hop(1GCN)
        z_mask = F.dropout(feat_mask, self.dropout, self.training)
        z_mask = torch.mm(z_mask, self.weight2_1)
        z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask) # I+A

        z_mask = self.act(z_mask)
        z_mask = F.dropout(z_mask, self.dropout, self.training)
        z_mask = torch.mm(z_mask, self.weight2_2)
        z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask) # I+A

        emb_mask = self.act(z_mask)

        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)

        g_mask = self.read(emb_mask, self.graph_neigh)
        g_mask = self.sigm(g_mask)


        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        ret_mask = self.disc(g_mask, emb_mask, emb)

        c = self.cluster_projector(z)
        c_mask = self.cluster_projector(z_mask)

        loss1 = self.compute_loss(z, z_mask)
        loss2 = self.compute_loss(z_mask, z)
        loss_mse = (loss1 + loss2) / 2

        return hiden_emb, h, ret, ret_a, ret_mask, c, c_mask, loss_mse
