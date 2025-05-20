import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import pickle as pkl
import sys
import os
import networkx as nx

# One-hot encoding cho nhãn
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

# Tính độ chính xác của mô hình
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # Dự đoán nhãn
    correct = preds.eq(labels).double()       # So sánh với nhãn thật
    correct = correct.sum()
    return correct / len(labels)

# Tính ROC-AUC score (yêu cầu sklearn)
def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("Thư viện sklearn chưa được cài đặt.")
    y_true = y_targets.cpu().numpy()
    y_true = encode_onehot(y_true)  # One-hot encoding
    y_pred = y_preds.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)

# Đọc file index chứa danh sách index test
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# Chuẩn hoá từng hàng trong ma trận (thường dùng cho features)
def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# Chuẩn hoá ma trận kề (adjacency matrix) theo công thức chuẩn của GCN
def adj_normalize(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])  # Thêm self-loop
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

# Tiền xử lý dữ liệu: chuẩn hoá adj và features
def preprocess_adj(adj, features):
    adj = adj_normalize(adj)
    features = row_normalize(features)
    return adj, features

# Chuyển scipy sparse matrix thành torch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Load dataset citation (Cora, Citeseer, Pubmed)
def load_citation(dataset_str="cora", porting_to_torch=True, data_path="data"):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, f"ind.{dataset_str.lower()}.{names[i]}"), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, f"ind.{dataset_str}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    # Xử lý riêng cho Citeseer do có test index không liên tục
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # Ghép features và labels của allx và tx
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    # Làm cho ma trận đối xứng
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    degree = np.sum(adj, axis=1)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # Chia tập train, val, test
    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally) - 500)
    idx_val = range(len(ally) - 500, len(ally))

    # Chuẩn hoá adj và features
    adj, features = preprocess_adj(adj, features)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)

    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)

    learning_type = "transductive"  # Kiểu học: xuyên suốt đồ thị (tập train & test cùng đồ thị)
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

# Wrapper để load toàn bộ dữ liệu đã xử lý
def data_loader(dataset, data_path="data", porting_to_torch=True):
    (adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type) = load_citation(dataset, porting_to_torch, data_path)
    train_adj = adj
    train_features = features
    return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type

# PairNorm: Chuẩn hoá theo chiều hàng (ngăn chặn over-smoothing trong GCN)
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        return x

# DropEdge: class hỗ trợ DropEdge và sampling dữ liệu
class DropEdge:
    def __init__(self, dataset, data_path="data"):
        self.dataset = dataset
        self.data_path = data_path
        (self.adj,
         self.train_adj,
         self.features,
         self.train_features,
         self.labels,
         self.idx_train,
         self.idx_val,
         self.idx_test,
         self.degree,
         self.learning_type) = data_loader(dataset, data_path, False)

        # Chuyển dữ liệu sang tensor
        self.features = torch.FloatTensor(self.features).float()
        self.train_features = torch.FloatTensor(self.train_features).float()
        self.labels_torch = torch.LongTensor(self.labels)
        self.idx_train_torch = torch.LongTensor(self.idx_train)
        self.idx_val_torch = torch.LongTensor(self.idx_val)
        self.idx_test_torch = torch.LongTensor(self.idx_test)

        # Lưu lại chỉ mục các sample dương và âm
        self.pos_train_idx = np.where(self.labels[self.idx_train] == 1)[0]
        self.neg_train_idx = np.where(self.labels[self.idx_train] == 0)[0]

        self.nfeat = self.features.shape[1]
        self.nclass = int(self.labels.max().item() + 1)

        self.trainadj_cache = {}  # Cache cho train adj
        self.adj_cache = {}       # Cache cho full adj
        self.degree_p = None

    # Tiền xử lý adj matrix
    def _preprocess_adj(self, normalization, adj, cuda):
        r_adj = adj_normalize(adj)
        r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
        if cuda:
            r_adj = r_adj.cuda()
        return r_adj

    # Tiền xử lý feature
    def _preprocess_fea(self, fea, cuda):
        return fea.cuda() if cuda else fea

    # Trả về toàn bộ adj và features (không drop edge)
    def stub_sampler(self, normalization, cuda):
        if normalization in self.trainadj_cache:
            r_adj = self.trainadj_cache[normalization]
        else:
            r_adj = self._preprocess_adj(normalization, self.train_adj, cuda)
            self.trainadj_cache[normalization] = r_adj
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea

    # Drop một phần cạnh (DropEdge) và tạo tập dữ liệu mới
    def randomedge_sampler(self, percent, normalization, cuda):
        if percent >= 1.0:
            return self.stub_sampler(normalization, cuda)
        nnz = self.train_adj.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix((self.train_adj.data[perm],
                               (self.train_adj.row[perm],
                                self.train_adj.col[perm])),
                              shape=self.train_adj.shape)
        r_adj = self._preprocess_adj(normalization, r_adj, cuda)
        fea = self._preprocess_fea(self.train_features, cuda)
        return r_adj, fea

    # Lấy dữ liệu cho test set
    def get_test_set(self, normalization, cuda):
        if self.learning_type == "transductive":
            return self.stub_sampler(normalization, cuda)
        else:
            if normalization in self.adj_cache:
                r_adj = self.adj_cache[normalization]
            else:
                r_adj = self._preprocess_adj(normalization, self.adj, cuda)
                self.adj_cache[normalization] = r_adj
            fea = self._preprocess_fea(self.features, cuda)
            return r_adj, fea

    # Trả về nhãn và index tập train/val/test
    def get_label_and_idxes(self, cuda):
        if cuda:
            return (self.labels_torch.cuda(),
                    self.idx_train_torch.cuda(),
                    self.idx_val_torch.cuda(),
                    self.idx_test_torch.cuda())
        return (self.labels_torch,
                self.idx_train_torch,
                self.idx_val_torch,
                self.idx_test_torch)
