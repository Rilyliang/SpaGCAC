import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.cluster import KMeans
from tqdm import tqdm
from ST_Model_1.preprocess import *
from ST_Model_1.model import *
from ST_Model_1.utils import *
import warnings
warnings.filterwarnings('ignore')

fix_seed(41)
datatype = '10X'
data_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
# data_list = ['151507']
for dataset in data_list:

    if dataset == '151669':
        n_clusters = 5
    elif dataset == '151670':
        n_clusters = 5
    elif dataset == '151671':
        n_clusters = 5
    elif dataset == '151672':
        n_clusters = 5
    else:
        n_clusters = 7

    file_fold = 'D:/scientific study/data/82_spatialLIBD_LIBD human dorsolateral pre-frontal cortex (DLPFC) spatial transcriptomics data generated with the 10x Genomics Visium platform/' + dataset + '/'
    adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    # add ground_truth
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs['ground_truth'] = df_meta_layer.values

    # df_meta = pd.read_csv(file_fold + '/clusters.csv', sep=',')
    # df_meta_layer = df_meta['Cluster']
    # adata.obs['ground_truth'] = df_meta_layer.values

    # filter out NA nodes
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    if 'highly_variable' not in adata.var.keys():
        preprocess(adata)
    if 'adj' not in adata.obsm.keys():
        if datatype in ['Stereo', 'Slide']:
            construct_interaction_KNN(adata)
        else:
            construct_interaction(adata, n_neighbors=3)
    if 'label_CSL' not in adata.obsm.keys():
        add_contrastive_label(adata)
    if 'label_CSL1' not in adata.obsm.keys():
        add_contrastive_label1(adata)
    if 'random_pos' not in adata.obsm.keys():
        random_pos(adata)
    if 'feat' not in adata.obsm.keys():
        get_feature(adata)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
    features_pos = torch.FloatTensor(adata.obsm['feat_pos'].copy()).to(device)

    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
    label_CSL1 = torch.FloatTensor(adata.obsm['label_CSL1']).to(device)
    adj = adata.obsm['adj']
    graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0])).to(device)  # 加自环
    identify_matrix = torch.FloatTensor(torch.eye(features.shape[0])).to(device)

    if datatype in ['Stereo', 'Slide']:
        print('Building sparse matrix ...')
        adj = preprocess_adj_sparse(adj).to(device)
    else:
        # standard version
        adj = preprocess_adj(adj)
        adj = torch.FloatTensor(adj).to(device)
    dim_input = features.shape[1]
    nnodes = features.shape[0]
    dim_output = 64
    nhid1 = 512
    nhid2 = 128


    if datatype in ['Stereo', 'Slide']:
        model = Encoder_sparse(nnodes, dim_input, nhid1, nhid2, dim_output, n_clusters, graph_neigh).to(device)
    else:
        model = Encoder(nnodes, dim_input, nhid1, nhid2, dim_output, n_clusters, graph_neigh).to(device)
    loss_CSL = nn.BCEWithLogitsLoss()

    loss_cluster = ClusterLoss(n_clusters, 0.5)
    loss_fn = InstanceLoss(0.07)

    criterion = nn.CrossEntropyLoss()

    learning_rate=0.001
    weight_decay=0.0
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    print('Begin to train ST data...')
    ari_max = 0.0
    epoch_max = 0.0
    epochs=600
    for epoch in tqdm(range(epochs)):
        model.train()

        features_a = permutation(features)
        hiden_feat, emb, ret, ret_a, ret_mask, c, c_mask, loss_mse= model(features, features_a, features_pos, adj, identify_matrix)

        loss_sl_1 = loss_CSL(ret, label_CSL)
        loss_sl_2 = loss_CSL(ret_a, label_CSL)
        loss_sl_3 = loss_CSL(ret_mask, label_CSL1)
        loss_feat = F.mse_loss(features, emb)
        loss =  5 * loss_feat + 0.1 * (loss_sl_1 + loss_sl_2 + loss_sl_3) + 0.1 * loss_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Optimization finished for ST data!")


    with torch.no_grad():
        model.eval()
        if datatype in ['Stereo', 'Slide']:
            emb_rec = self.model(features, features_a, features_pos, adj, identify_matrix)[1]
            emb_rec = F.normalize(emb_rec, p=2, dim=1).detach().cpu().numpy()
        else:
            emb_rec = model(features, features_a, features_pos, adj, identify_matrix)[1].detach().cpu().numpy()
        adata.obsm['emb'] = emb_rec

    # print(adata)
    radius = 50
    tool = 'mclust' # mclust, leiden, and louvain
    # clustering

    if tool == 'mclust':
       clustering(adata, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
       clustering(adata, n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)

    # calculate metric ARI
    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
    adata.uns['ARI'] = ARI
    adata.uns['NMI'] = NMI

    print('Dataset:', dataset)
    print('ARI:', ARI)
    print('NMI:', NMI)
