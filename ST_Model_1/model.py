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
        # z = torch.mm(adj, z)
        z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z) # I+A

        z = self.act(z)
        z = F.dropout(z, self.dropout, self.training)
        z = torch.mm(z, self.weight2_2)
        # z = torch.mm(adj, z) # A
        z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z) # I+A

        # z = self.act(z)
        # z = F.dropout(z, self.dropout, self.training)
        # z = torch.mm(z, self.weight3_2)
        # # z = torch.mm(adj, z)
        # z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z)
        #
        # z = self.act(z)
        # z = F.dropout(z, self.dropout, self.training)
        # z = torch.mm(z, self.weight2_2)
        # # z = torch.mm(adj, z)
        # z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z)


        # 解码基因表达
        hiden_emb = z #原始数据经过GCN得到的潜在表达hiden_emb
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h) #解码后的表达（原始数据得到的）h

        emb = self.act(z) #对中间得到的潜在表达使用激活函数

        # 1-hop(1GCN)
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight2_1)
        # z_a = torch.mm(adj, z_a) # A
        z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a) # I+A

        z_a = self.act(z_a)
        z_a = F.dropout(z_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight2_2)
        # z_a = torch.mm(adj, z_a) # A
        z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a)  # I+A

        # z_a = self.act(z_a)
        # z_a = F.dropout(z_a, self.dropout, self.training)
        # z_a = torch.mm(z_a, self.weight3_2)
        # # z_a = torch.mm(adj, z_a)
        # z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a)  # I+A1
        #
        # z_a = self.act(z_a)
        # z_a = F.dropout(z_a, self.dropout, self.training)
        # z_a = torch.mm(z_a, self.weight2_2)
        # # z_a = torch.mm(adj, z_a)
        # z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a)  # I+A1

        emb_a = self.act(z_a)

        # 1-hop(1GCN)
        z_mask = F.dropout(feat_mask, self.dropout, self.training)
        z_mask = torch.mm(z_mask, self.weight2_1)
        # z_mask = torch.mm(adj, z_mask) # A
        z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask) # I+A

        z_mask = self.act(z_mask)
        z_mask = F.dropout(z_mask, self.dropout, self.training)
        z_mask = torch.mm(z_mask, self.weight2_2)
        # z_mask = torch.mm(adj, z_mask) # A
        z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask) # I+A

        # z_mask = self.act(z_mask)
        # z_mask = F.dropout(z_mask, self.dropout, self.training)
        # z_mask = torch.mm(z_mask, self.weight3_2)
        # # z_mask = torch.mm(adj, z_mask)
        # z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask)
        #
        # z_mask = self.act(z_mask)
        # z_mask = F.dropout(z_mask, self.dropout, self.training)
        # z_mask = torch.mm(z_mask, self.weight2_2)
        # # z_mask = torch.mm(adj, z_mask)
        # z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask)

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
    
class Encoder_sparse(Module):
    """
    Sparse version of Encoder
    """
    def __init__(self, nnodes, in_features, nhid1, nhid2, out_features, n_clusters, graph_neigh, dropout=0.1, act=F.relu):
        super(Encoder_sparse, self).__init__()
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
        # z = torch.mm(adj, z) #A
        z = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z) #I+A


        # 解码基因表达
        hiden_emb = z  # 原始数据经过GCN得到的潜在表达hiden_emb
        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)  # 解码后的表达（原始数据得到的）h

        emb = self.act(z)  # 对中间得到的潜在表达使用激活函数

        # 1-hop(1GCN)
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight2_1)
        # z_a = torch.mm(adj, z_a) # A
        z_a = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_a)  # I+A1


        emb_a = self.act(z_a)

        # 1-hop(1GCN)
        z_mask = F.dropout(feat_mask, self.dropout, self.training)
        z_mask = torch.mm(z_mask, self.weight2_1)
        # z_mask = torch.mm(adj, z_mask) # A
        z_mask = torch.mm((self.lamda1 * identify_matrix + self.lamda2 * adj), z_mask) #I+A


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

class Encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.0, act=F.relu):
        super(Encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dim3 = 32
        self.act = act
        self.dropout = dropout
        
        #self.linear1 = torch.nn.Linear(self.dim_input, self.dim_output)
        #self.linear2 = torch.nn.Linear(self.dim_output, self.dim_input)
        
        #self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim_output))
        #self.weight1_de = Parameter(torch.FloatTensor(self.dim_output, self.dim_input))
        
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight3_en = Parameter(torch.FloatTensor(self.dim2, self.dim3))
        
        self.weight1_de = Parameter(torch.FloatTensor(self.dim3, self.dim2))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight3_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
      
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
        torch.nn.init.xavier_uniform_(self.weight3_en)
        torch.nn.init.xavier_uniform_(self.weight3_de)
        
    def forward(self, x):
        x = F.dropout(x, self.dropout, self.training)
        
        #x = self.linear1(x)
        #x = self.linear2(x)
        
        #x = torch.mm(x, self.weight1_en)
        #x = torch.mm(x, self.weight1_de)
        
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight3_en)
        
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        x = torch.mm(x, self.weight3_de)
        
        return x
    
class Encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(Encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
        
    def forward(self):
        x = self.M
        
        return x 


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):###定义ClusterLoss这个类的基本参数和方法
        super(ClusterLoss, self).__init__()
        self.class_num = class_num###类别数目，比如deng是10
        self.temperature = temperature###温度参数=1.0
        self.mask = self.mask_correlated_clusters(class_num)###定义mask的方式
        self.criterion = nn.CrossEntropyLoss(reduction="sum")###定义损失标准采用交叉熵
        self.similarity_f = nn.CosineSimilarity(dim=2)###定义相似性采用余弦相似度

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))###生成(N, N)的全1矩阵
        mask = mask.fill_diagonal_(0)###对角线元素置0
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()###转成bool型变量，即其中的1变成True
        return mask

    def forward(self, c_i, c_j):##对每一批，c_i是256个10维的向量，
        p_i = c_i.sum(0).view(-1)###把这一批256个求和，得到一个总的p_i，是一个10维的向量
        p_i /= p_i.sum()###p_i.sum()=256,所以p_i这里是求平均,是一个10维的向量
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()###log(p_i)求得是以e为底的ln(p_i),
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()####转置，变成10*256的
        c_j = c_j.t()###转置，变成10*256的
        N = 2 * self.class_num###N=20
        c = torch.cat((c_i, c_j), dim=0)##拼接，变成20*256的

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature###20*20的
        sim_i_j = torch.diag(sim, self.class_num)###取矩阵的主对角线元素 1*10
        sim_j_i = torch.diag(sim, -self.class_num) #取矩阵的副对角线元素 1*10

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) #20 * 1
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
