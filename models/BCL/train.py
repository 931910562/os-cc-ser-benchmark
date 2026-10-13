# BCL training and evaluation on pre-extracted features.

import os
import csv
import random
import warnings
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from hsic import hsic_regular, hsic_regular0_block

warnings.filterwarnings("ignore")


# ==================== Argument Parser ====================
import argparse

parser = argparse.ArgumentParser(
    description="BCL-OSDA Training Script with Detailed Logs"
)

parser.add_argument(
    '--data_root',
    type=str,
    default='/seu_nvme/home/220242615/code/code/BCL_OSDA-main/BCL_OSDA-main/model/egemaps',
    help='Path of feature .mat files',
)
parser.add_argument(
    '--source_domain', type=str, default='C', help='Source domain: C/E/M/Me'
)
parser.add_argument(
    '--target_domain', type=str, default='E', help='Target domain: C/E/M/Me'
)


parser.add_argument(
    '--source_cls_idx',
    type=int,
    nargs='+',
    default=[0, 1, 2],
    help='Known source class indices',
)
parser.add_argument(
    '--target_cls_idx',
    type=int,
    nargs='+',
    default=[0, 1, 2, 3, 4],
    help='Target class indices',
)


parser.add_argument(
    '--n_classes',
    type=int,
    default=4,
    help='Number of classifier outputs: known classes + unknown',
)

parser.add_argument(
    '--lambda1',
    type=float,
    nargs='+',
    default=[0.008],
    help='Unknown CE loss coefficient values',
)
parser.add_argument(
    '--lambda2',
    type=float,
    nargs='+',
    default=[0.28],
    help='Class-wise HSIC coefficient values',
)
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Batch size. None is not used for full-batch loaders here.',
)
parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
parser.add_argument(
    '--feature_dim', type=int, default=1024, help='Input feature dimension'
)
parser.add_argument(
    '--Exptimes', type=int, default=2, help='Number of repeated experiments'
)
parser.add_argument(
    '--TarEnt_step', type=int, default=50, help='Start epoch for target adaptation loss'
)
parser.add_argument(
    '--save_path',
    type=str,
    default='/seu_nvme/home/220242615/code/code/BCL_OSDA-main/BCL_OSDA-main/model/egemaps',
    help='Path to save logs and models',
)
parser.add_argument('--gpu', type=int, default=0, help='CUDA device id')
parser.add_argument(
    '--kmeans_clusters',
    type=int,
    default=3,
    help='KMeans clusters for target entropy grouping',
)
parser.add_argument('--seed', type=int, default=1024, help='Random seed')

args = parser.parse_args()


# ==================== GPU ====================
torch.cuda.set_device(args.gpu)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


# ==================== Utils ====================
def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_mat_path(domain):
    data_dir = args.data_root
    if domain == 'C':
        file_name = 'C.mat'
    elif domain == 'E':
        file_name = 'E.mat'
    elif domain == 'M':
        file_name = 'M.mat'
    elif domain == 'Me':
        file_name = 'Me.mat'
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return os.path.join(data_dir, file_name)


def path_maker(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print("Created folder:", file_path)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def tensor_to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)


def map_target_label_to_osda(lab_t):
    """Map target labels to the open-set label space.

    Labels below unknown_id are retained. Labels greater than or equal to
    unknown_id are mapped to unknown_id = n_classes - 1.

    With the default n_classes=4, labels 0, 1 and 2 remain unchanged,
    and labels 3 and above are mapped to 3.
    """
    unknown_id = args.n_classes - 1
    lab_t = torch.where(lab_t >= unknown_id, torch.full_like(lab_t, unknown_id), lab_t)
    return lab_t


# ==================== Dataset ====================
class ImageSet_dataset(Dataset):
    """
    Dataset for domain adaptation using pre-extracted .mat features.

    Expected final format after loading:
        features: shape [N, D]
        labels:   shape [N]

    This version is compatible with MATLAB v7.3 .mat files loaded by h5py.
    It also automatically handles the case where features are stored as [D, N].
    """

    def __init__(self, mat_path, partial_cls_idx=None):
        # Load features and labels from MATLAB v7.3 .mat file
        with h5py.File(mat_path, 'r') as f:
            features = np.array(f['features'])
            labels = np.array(f['labels']).squeeze().astype(np.int64)

        # Ensure labels are a one-dimensional array with shape [N]
        labels = labels.reshape(-1)

        # =====================================================
        # Important fix:
        # Some .mat files store features as [D, N],
        # for example [2048, 12923], while labels have shape [12923].
        # In this case, features must be transposed to [N, D].
        # =====================================================
        if features.ndim == 2:
            if features.shape[0] == len(labels):
                # Features are already in the correct format: [N, D]
                pass
            elif features.shape[1] == len(labels):
                # Features are stored as [D, N], so transpose them to [N, D]
                print(
                    "[INFO] Transpose features: {} -> {}".format(
                        features.shape, features.T.shape
                    )
                )
                features = features.T
            else:
                raise ValueError(
                    "The number of samples in features and labels does not match: "
                    "features.shape={}, labels.shape={}".format(
                        features.shape, labels.shape
                    )
                )
        else:
            raise ValueError(
                "features should be a 2D matrix with shape [N, D] or [D, N], "
                "but got shape={}".format(features.shape)
            )

        # Convert numpy arrays to PyTorch tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        # =====================================================
        # Select samples belonging to the specified class indices
        # This is used for source_cls_idx or target_cls_idx filtering.
        # =====================================================
        if partial_cls_idx is not None:
            if isinstance(partial_cls_idx, str):
                partial_cls_idx = [int(x) for x in partial_cls_idx.split()]

            mask = torch.zeros(len(labels), dtype=torch.bool)

            for cls in partial_cls_idx:
                mask |= labels == int(cls)

            features = features[mask]
            labels = labels[mask]

        self.features = features
        self.labels = labels

        # Print dataset information for debugging
        print("[Dataset] {}".format(mat_path))
        print("  features:", self.features.shape)
        print("  labels  :", self.labels.shape)
        print("  classes :", sorted(set(self.labels.numpy().tolist())))

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def Make_Loader(mat_path, Batch_size=None, partial_cls_idx=None):
    data_set = ImageSet_dataset(mat_path, partial_cls_idx)
    if Batch_size is None:
        Batch_size = len(data_set)
    new_loader = DataLoader(data_set, batch_size=Batch_size, shuffle=False)
    return new_loader


# ==================== Model ====================
class C(nn.Module):
    def __init__(self, input_dim, conv_dim_1, conv_dim_2):
        super(C, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, conv_dim_1),
            nn.BatchNorm1d(conv_dim_1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(conv_dim_1, conv_dim_2),
            nn.BatchNorm1d(conv_dim_2),
            nn.Tanh(),
        )
        self.fc3 = nn.Linear(conv_dim_2, args.n_classes)

    def forward(self, x):
        z_1 = self.fc1(x)
        z_2 = self.fc2(z_1)
        y = self.fc3(z_2)
        p = F.softmax(y, dim=1)
        return z_1, z_2, p


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)


def reset_grad():
    c.zero_grad()


# ==================== Metrics ====================
def classification_accuracy_train():
    with torch.no_grad():
        iter_loader_s = iter(source_loader)
        X_s, lab_s = next(iter_loader_s)
        X_s, lab_s = to_var(X_s), to_var(lab_s).long().squeeze()

        _, _, output = c(X_s)
        pred = output.data.max(1)[1]
        correct = pred.eq(lab_s.data).cpu().float()

        return correct.mean().item()


def classification_accuracy():

    with torch.no_grad():
        X_t, lab_t = next(iter(target_loader))
        X_t, lab_t = X_t.cuda(), lab_t.cuda()

        lab_t = map_target_label_to_osda(lab_t)

        _, _, output = c(X_t)
        pred = output.data.max(1)[1]

        acc_per_class = torch.zeros(args.n_classes).cuda()

        for i in range(args.n_classes):
            class_mask = lab_t == i
            class_num = torch.sum(class_mask).float()
            if class_num > 0:
                acc_per_class[i] = (
                    torch.sum((pred == lab_t) & class_mask).float() / class_num
                )
            else:
                acc_per_class[i] = 0.0

        OS_star = torch.mean(acc_per_class[0 : args.n_classes - 1])
        UNK = acc_per_class[args.n_classes - 1]
        OS = torch.mean(acc_per_class)
        HOS = 2 * OS_star * UNK / (OS_star + UNK + 1e-5)

        return (
            HOS.detach().cpu().item(),
            OS.detach().cpu().item(),
            OS_star.detach().cpu().item(),
            UNK.detach().cpu().item(),
            acc_per_class.detach().cpu().numpy(),
        )


def Entropy(x):
    epsilon = 1e-5
    entropy = -x * torch.log(x + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def Pred_Entropy_loss(pred_t):
    num_sam = pred_t.shape[0]
    Entropy_val = -(pred_t.mul(pred_t.log() + 1e-4)).sum()
    return Entropy_val / num_sam


def One_Hot(lab, num_cls):
    num_sam = lab.shape[0]
    lab_mat = torch.zeros([num_sam, num_cls]).cuda().scatter_(1, lab.unsqueeze(1), 1)
    return lab_mat


# ==================== Loaders ====================
source_path = get_mat_path(args.source_domain)
target_path = get_mat_path(args.target_domain)

source_loader = Make_Loader(source_path, partial_cls_idx=args.source_cls_idx)
target_loader = Make_Loader(target_path, partial_cls_idx=args.target_cls_idx)


# ==================== Training ====================
Detail_result = ()
Max_ACC = np.zeros((args.Exptimes + 1, 3))
Total_Result = np.zeros((8, args.epochs, args.Exptimes))

for lambda2_i in range(len(args.lambda2)):
    lambda00 = args.lambda2[lambda2_i]

    for lambda1_i in range(len(args.lambda1)):
        lambda0 = args.lambda1[lambda1_i]

        lambda_dir_name = f"l1{lambda0:.4f}_l2{lambda00:.4f}"
        base_path = os.path.join(
            args.save_path,
            f"{args.source_domain}to{args.target_domain}",
            lambda_dir_name,
        )
        path_maker(base_path)

        for Domain_iter in range(1):
            set_seed(args.seed)

            source_domain = args.source_domain
            target_domain = args.target_domain

            source_path = get_mat_path(source_domain)
            target_path = get_mat_path(target_domain)

            source_loader = Make_Loader(
                source_path, partial_cls_idx=args.source_cls_idx
            )
            target_loader = Make_Loader(
                target_path, partial_cls_idx=args.target_cls_idx
            )

            Total_Result = np.zeros((8, args.epochs, args.Exptimes))

            Exp_iter = 0
            feature_dim = args.feature_dim

            while Exp_iter < args.Exptimes:
                c = C(feature_dim, 1024, 512).to(device)
                c.apply(weights_init)

                lr = args.lr
                beta1, beta2 = 0.9, 0.999
                c_solver = optim.Adam(
                    c.parameters(), lr, [beta1, beta2], weight_decay=0.01
                )
                criterionQ_dis = nn.NLLLoss().cuda()

                # Result directory
                # 0 CE_loss
                # 1 TarEnt_loss
                # 2 Total_loss
                # 3 Unknown_CE_loss
                # 4 train_acc
                # 5 HOS
                # 6 OS_star
                # 7 UNK
                result = np.zeros((8, args.epochs))

                Exp_start_time = datetime.now()
                print(
                    f"************ {args.source_domain}->{args.target_domain}: Experiment {Exp_iter + 1} Start {Exp_start_time} ************"
                )

                # ==================== Epoch CSV Log ====================
                epoch_log_path = os.path.join(base_path, f"epoch_log_exp{Exp_iter}.csv")
                class_acc_headers = [f"acc_class_{i}" for i in range(args.n_classes)]

                with open(epoch_log_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "epoch",
                            "CE_loss",
                            "TarEnt_loss",
                            "Total_loss",
                            "Unknown_CE_loss",
                            "I_xy",
                            "I_xz",
                            "I_c",
                            "pseudo_unknown_num",
                            "pseudo_known_num",
                            "train_acc",
                            "HOS",
                            "OS",
                            "OS_star",
                            "UNK",
                            "lr",
                            "epoch_time_sec",
                        ]
                        + class_acc_headers
                    )

                best_hos = -1.0
                best_epoch = -1
                best_os = 0.0
                best_os_star = 0.0
                best_unk = 0.0

                for step in range(args.epochs):
                    epoch_time_start = datetime.now()

                    c.train()

                    iter_loader_s = iter(source_loader)
                    iter_loader_t = iter(target_loader)

                    X_s, lab_s = next(iter_loader_s)
                    X_t, lab_t = next(iter_loader_t)

                    X_s, lab_s = Variable(X_s), Variable(lab_s)
                    X_s, lab_s = X_s.cuda(), lab_s.cuda()

                    X_t, lab_t = Variable(X_t), Variable(lab_t)
                    X_t, lab_t = X_t.cuda(), lab_t.cuda()

                    lab_t_osda = map_target_label_to_osda(lab_t)
                    lab_s0 = lab_s.cpu().numpy()

                    reset_grad()

                    h_s, h_s2, pred_s = c(X_s)
                    h_t, h_t2, pred_t = c(X_t)

                    n = lab_s.shape[0]
                    m = lab_t.shape[0]

                    fs_norm = (torch.norm(h_s2, 2, dim=1).repeat(h_s2.shape[1], 1)).T
                    ft_norm = (torch.norm(h_t2, 2, dim=1).repeat(h_t2.shape[1], 1)).T

                    fs = h_s2 / (fs_norm + 1e-8)
                    ft = h_t2 / (ft_norm + 1e-8)

                    # Source one-hot labels
                    ys_one_hot = torch.zeros(n, args.n_classes).cuda()
                    ys_one_hot = ys_one_hot.scatter_(1, lab_s.reshape(n, 1), 1)

                    # Target pseudo labels
                    yt_one_hot = torch.zeros(m, args.n_classes).cuda()

                    # ============ pseudo unknown split by target prediction entropy + KMeans ============
                    entropy_t = Entropy(pred_t).reshape(1, -1).detach()
                    entropy_np = entropy_t.cpu().numpy().T

                    cluster_num = min(args.kmeans_clusters, len(entropy_np))
                    cluster = KMeans(n_clusters=cluster_num, random_state=0).fit(
                        entropy_np
                    )
                    centroid = cluster.cluster_centers_

                    high_entropy_cluster = np.argmax(centroid)
                    index_u = cluster.labels_ == high_entropy_cluster
                    index_u = torch.from_numpy(index_u).squeeze().cuda().bool()

                    pred_t_NG = pred_t.detach()
                    pse_t = pred_t_NG.data.max(1)[1]

                    yt_one_hot = yt_one_hot.scatter_(1, pse_t.reshape(m, 1), 1)

                    unknown_id = args.n_classes - 1

                    h_k1 = (~index_u) & (pse_t != unknown_id)

                    # ============ class-wise HSIC between source known and target pseudo-known ============
                    I_c = torch.zeros((), device=device)

                    for i in range(args.n_classes - 1):
                        index_pred_i = pse_t == i
                        index_t = h_k1 & index_pred_i
                        index_s = lab_s0 == i

                        if not index_t.any():
                            continue

                        m1 = fs[index_s, :].shape[0]
                        if m1 == 0:
                            continue

                        X = torch.cat((fs[index_s, :], ft[index_t, :]), dim=0)

                        D = torch.zeros(X.shape[0], 2).cuda()
                        D[0:m1, 0] = 1
                        D[m1 : X.shape[0], 1] = 1

                        I_c = I_c + hsic_regular(X, D)

                    # ============ Unknown pseudo labels ============
                    x_u = ft[index_u, :]
                    u = x_u.shape[0]

                    y_u = torch.zeros(u, args.n_classes).cuda()
                    if u > 0:
                        y_u[:, unknown_id] = 1
                    l_u = unknown_id * torch.ones(u).long().cuda()

                    # ============ Global HSIC terms ============
                    if u != 0:
                        X0 = torch.cat((fs, ft[h_k1, :], x_u), dim=0)
                        Y0 = torch.cat((ys_one_hot, yt_one_hot[h_k1, :], y_u), dim=0)

                        n_all = Y0.shape[0]
                        class_count = torch.sum(Y0, dim=0).clamp_min(1.0)
                        por = (1 / class_count / args.n_classes).reshape(-1, 1)
                        a = torch.mm(Y0, por).cuda()

                        fx = torch.cat((X_s, X_t[h_k1, :], X_t[index_u, :]), dim=0)

                        I_xy = n_all * hsic_regular0_block(X0, Y0, a, block_size=128)
                        I_xz = n_all * hsic_regular0_block(X0, fx, a, block_size=128)

                    else:
                        X0 = torch.cat((fs, ft[h_k1, :]), dim=0)
                        Y0 = torch.cat((ys_one_hot, yt_one_hot[h_k1, :]), dim=0)

                        n_all = Y0.shape[0]
                        class_count = torch.sum(Y0, dim=0).clamp_min(1.0)
                        por = (1 / class_count / (args.n_classes - 1)).reshape(-1, 1)
                        por[unknown_id] = 0
                        a = torch.mm(Y0, por).cuda()

                        fx = torch.cat((X_s, X_t[h_k1, :], X_t[index_u, :]), dim=0)

                        I_xy = n_all * hsic_regular0_block(X0, Y0, a, block_size=128)
                        I_xz = n_all * hsic_regular0_block(X0, fx, a, block_size=128)

                    # ==========================================================
                    # Loss Part
                    # ==========================================================
                    CE_loss = criterionQ_dis(torch.log(pred_s + 1e-4), lab_s)

                    if step <= (args.TarEnt_step - 1):
                        Tar_Ent_loss = torch.zeros(1).squeeze(0).cuda()
                        Unknown_CE_loss = torch.zeros(1).squeeze(0).cuda()
                    else:
                        if u == 0:
                            Unknown_CE_loss = torch.zeros(1).squeeze(0).cuda()
                            Tar_Ent_loss = (
                                0.8e-2 * I_xz - 0.8e-3 * I_xy + lambda00 * I_c
                            )
                        else:
                            Unknown_CE_loss = criterionQ_dis(
                                torch.log(pred_t[index_u, :] + 1e-4), l_u
                            )
                            Tar_Ent_loss = (
                                0.8e-2 * I_xz
                                - 0.8e-3 * I_xy
                                + lambda00 * I_c
                                + lambda0 * Unknown_CE_loss
                            )

                    c_loss = CE_loss + Tar_Ent_loss

                    c_loss.backward()
                    c_solver.step()

                    # ==========================================================
                    # Test and log
                    # ==========================================================
                    c.eval()

                    Test_start_time = datetime.now()
                    print(
                        '========================== %1s | Testing start! ==========================='
                        % (Test_start_time)
                    )

                    HOS, OS, OS_star, Unk, acc_per_class = classification_accuracy()
                    train_acc = classification_accuracy_train()

                    result[:, step] = [
                        tensor_to_float(CE_loss),
                        tensor_to_float(Tar_Ent_loss),
                        tensor_to_float(c_loss),
                        tensor_to_float(Unknown_CE_loss),
                        train_acc,
                        HOS,
                        OS_star,
                        Unk,
                    ]

                    epoch_time_end = datetime.now()
                    seconds = (epoch_time_end - epoch_time_start).seconds
                    minutes = seconds // 60
                    second = seconds % 60
                    hours = minutes // 60
                    minute = minutes % 60

                    CE_loss_value = tensor_to_float(CE_loss)
                    TarEnt_loss_value = tensor_to_float(Tar_Ent_loss)
                    Total_loss_value = tensor_to_float(c_loss)
                    Unknown_CE_loss_value = tensor_to_float(Unknown_CE_loss)
                    I_xy_value = tensor_to_float(I_xy)
                    I_xz_value = tensor_to_float(I_xz)
                    I_c_value = tensor_to_float(I_c)

                    pseudo_unknown_num = int(index_u.sum().detach().cpu().item())
                    pseudo_known_num = int(h_k1.sum().detach().cpu().item())

                    print(
                        '====================== %1s→%1s: Experiment %1s Epoch %1s ================='
                        % (
                            args.source_domain,
                            args.target_domain,
                            Exp_iter + 1,
                            step + 1,
                        )
                    )

                    print('Train accuracy: {:.4f}'.format(train_acc))
                    print('CE_Loss: {:.6f}'.format(CE_loss_value))
                    print('TarEnt_Loss: {:.6f}'.format(TarEnt_loss_value))
                    print('Unknown_CE_Loss: {:.6f}'.format(Unknown_CE_loss_value))
                    print('Total_Loss: {:.6f}'.format(Total_loss_value))
                    print(
                        'I_xy: {:.6f} | I_xz: {:.6f} | I_c: {:.6f}'.format(
                            I_xy_value, I_xz_value, I_c_value
                        )
                    )
                    print(
                        'HOS: {:.4f} | OS: {:.4f} | OS*: {:.4f} | UNK: {:.4f}'.format(
                            HOS, OS, OS_star, Unk
                        )
                    )
                    print('Class Acc:', acc_per_class)
                    print(
                        'Pseudo known num: {} | Pseudo unknown num: {}'.format(
                            pseudo_known_num, pseudo_unknown_num
                        )
                    )
                    print(
                        'Current epoch time cost: %1s Hour %1s Minutes %1s Seconds'
                        % (hours, minute, second)
                    )

                    with open(epoch_log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                step + 1,
                                CE_loss_value,
                                TarEnt_loss_value,
                                Total_loss_value,
                                Unknown_CE_loss_value,
                                I_xy_value,
                                I_xz_value,
                                I_c_value,
                                pseudo_unknown_num,
                                pseudo_known_num,
                                train_acc,
                                HOS,
                                OS,
                                OS_star,
                                Unk,
                                lr,
                                seconds,
                            ]
                            + list(acc_per_class)
                        )

                    if HOS > best_hos:
                        best_hos = HOS
                        best_epoch = step + 1
                        best_os = OS
                        best_os_star = OS_star
                        best_unk = Unk

                        best_model_save_path = os.path.join(
                            base_path, f"c_model_exp{Exp_iter}_best.pth"
                        )
                        torch.save(c.state_dict(), best_model_save_path)

                    if max(result[5, :]) == 1:
                        print('Reach HOS 1.0 at Epoch %1s !' % (step + 1))
                        break

                # ==================== Save result for one experiment ====================
                seconds_total = (epoch_time_end - Exp_start_time).seconds
                minutes_total = seconds_total // 60
                second_total = seconds_total % 60
                hours_total = minutes_total // 60
                minute_total = minutes_total % 60

                index = np.argmax(result[5, :])
                Max_ACC[Exp_iter, 0] = result[5, index]  # best HOS
                Max_ACC[Exp_iter, 1] = result[6, index]  # OS_star at best HOS
                Max_ACC[Exp_iter, 2] = result[7, index]  # UNK at best HOS

                Total_Result[:, :, Exp_iter] = result

                model_save_path = os.path.join(
                    base_path, f"c_model_exp{Exp_iter}_last.pth"
                )
                torch.save(c.state_dict(), model_save_path)

                total_result_save_path = os.path.join(
                    base_path, f"Total_Result_exp{Exp_iter}.npy"
                )
                np.save(total_result_save_path, Total_Result[:, :, Exp_iter])

                max_acc_save_path = os.path.join(
                    base_path, f"Max_ACC_exp{Exp_iter}.txt"
                )
                np.savetxt(
                    max_acc_save_path,
                    Max_ACC[Exp_iter, :].reshape(1, -1),
                    fmt="%.5f",
                    delimiter='\t',
                    header="Best_HOS\tOS_star_at_best_HOS\tUNK_at_best_HOS",
                )

                summary_log_file_path = os.path.join(
                    base_path, f"summary_log_exp{Exp_iter}.txt"
                )
                with open(summary_log_file_path, 'w') as f:
                    f.write(f"Training summary for experiment {Exp_iter}\n")
                    f.write(f"Start Time: {Exp_start_time}\n")
                    f.write(f"End Time: {datetime.now()}\n")
                    f.write(f"Best epoch: {index + 1}\n")
                    f.write(f"Best HOS: {Max_ACC[Exp_iter, 0]}\n")
                    f.write(f"OS_star at best HOS: {Max_ACC[Exp_iter, 1]}\n")
                    f.write(f"UNK at best HOS: {Max_ACC[Exp_iter, 2]}\n")
                    f.write(
                        f"Best model according to online HOS: {os.path.join(base_path, f'c_model_exp{Exp_iter}_best.pth')}\n"
                    )
                    f.write(f"Last model path: {model_save_path}\n")
                    f.write(f"Epoch CSV log path: {epoch_log_path}\n")
                    f.write(f"Total_Result path: {total_result_save_path}\n")
                    f.write(f"Max_ACC path: {max_acc_save_path}\n")
                    f.write("\nResult row meanings:\n")
                    f.write("result[0] = CE_loss\n")
                    f.write("result[1] = TarEnt_loss\n")
                    f.write("result[2] = Total_loss\n")
                    f.write("result[3] = Unknown_CE_loss\n")
                    f.write("result[4] = train_acc\n")
                    f.write("result[5] = HOS\n")
                    f.write("result[6] = OS_star\n")
                    f.write("result[7] = UNK\n")

                print('Current Max HOS: {}'.format(max(result[5, :])))
                print('Best epoch: {}'.format(index + 1))
                print(
                    'Best HOS / OS* / UNK: {:.5f} / {:.5f} / {:.5f}'.format(
                        Max_ACC[Exp_iter, 0], Max_ACC[Exp_iter, 1], Max_ACC[Exp_iter, 2]
                    )
                )
                print('Starting Time: {}'.format(Exp_start_time))
                print("Finishing Time: {}".format(epoch_time_end))
                print(
                    'Total Time Cost: %1s Hour %1s Minutes %1s Seconds'
                    % (hours_total, minute_total, second_total)
                )
                print(
                    "************ %1s→%1s: %1s End Experiment %1s training ************"
                    % (source_domain, target_domain, epoch_time_end, Exp_iter + 1)
                )

                Exp_iter += 1

            Max_ACC[args.Exptimes, :] = np.mean(Max_ACC[0 : args.Exptimes, :], axis=0)
            print(Max_ACC)

            avg_max_acc_path = os.path.join(base_path, "Max_ACC_avg.txt")
            np.savetxt(
                avg_max_acc_path,
                Max_ACC[args.Exptimes, :].reshape(1, -1),
                fmt="%.5f",
                delimiter='\t',
                header="Avg_Best_HOS\tAvg_OS_star_at_best_HOS\tAvg_UNK_at_best_HOS",
            )

            print(
                f"[INFO] All experiments finished. Average Max_ACC saved at: {avg_max_acc_path}"
            )

            Detail_result = Detail_result + (Total_Result,)
