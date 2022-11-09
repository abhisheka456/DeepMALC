import argparse
import os
import random as rn
from DE import DeepEmbedding

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from GridShiftPP import *
import optuna as opt
import math 
from sklearn.metrics import silhouette_score
from  sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use(['seaborn-white', 'seaborn-paper'])
sns.set_context("paper", font_scale=1.3)
import pandas as pd
import numpy as np
import sys
import torch
import diptest
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde



from sklearn import metrics
from sklearn import mixture
from hdbscan.validity import validity_index
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

from scipy.optimize import linear_sum_assignment as linear_assignment
from time import time





np.set_printoptions(threshold=sys.maxsize)

matplotlib.use('agg')



def cluster_manifold_in_embedding(hl, y, label_names=None):
    # find manifold on autoencoded embedding
    md = float(args.umap_min_dist)
    de = DeepEmbedding(
        random_shuffle=False,
        n_components=args.n_components,
        dataset_name=args.dataset,
        tsne_perplexity=30.0,
        umap_n_neighbors=args.umap_neighbors,
        min_dist=md,
        batch_size=args.batch_size,
        num_pre_epochs=args.num_pre_epochs,
        num_ae_epochs=args.num_ae_epochs,
        num_recursive_tsne_epochs=args.num_recursive_tsne_epochs,
        num_recursive_umap_epochs=args.num_recursive_umap_epochs,
        plot_init = True,
        load_weight_emb=args.load_weight_emb,
        )
    hle = de.fit_transform(hl)

    if args.cluster == 'GMM':
        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=args.n_clusters,
            random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
    elif args.cluster == 'KM':
        km = KMeans(
            init='k-means++',
            n_clusters=args.n_clusters,
            random_state=0,
            n_init=20)
        y_pred = km.fit_predict(hle)
    elif args.cluster == 'SC':
        sc = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)
    elif args.cluster == 'GS':

        h, knn_idx, knn_dist, rho, core_card = est_h(hle,args.p)
        gs = GridShiftPP(bandwidth=h, iterations=200)
        y_pred, centers = gs.fit_predict(hle)
        y_pred, centers = get_reduce_noise(hle, y_pred, centers)
        y_pred, centers = merge_cluster(hle, h, y_pred, knn_idx, knn_dist, rho, core_card)

        

    y_pred = np.asarray(y_pred)
    y = np.asarray(y)
    dbi = np.round(validity_index(hle.astype(np.double), y_pred),5)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | " + " on autoencoded embedding with " + args.cluster)
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print(dbi)
    print(y_pred.max()+1)
    print(h)
    print(args.p)
    print(args.tol)
    print('=' * 80)

    if args.visualize:
        plot(hle, y, 'n2d', label_names)
        y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
        plot(hle, y_pred_viz, 'n2d-predicted', label_names)

    return y_pred, acc, nmi, ari, dbi, h


def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    x, y = linear_assignment(w.max() - w)
    ind = np.array(list(zip(x, y)))
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



def est_h(X,p):
    nb_el = len(X)
    effective_nb_el = int(nb_el / 14)
    # Use kNN to get neighbours + idx
    vectors = X.astype(np.float32)
    nbrs = NearestNeighbors(n_neighbors=effective_nb_el, algorithm="auto", n_jobs=1, metric='chebyshev').fit(vectors)
    knn_dist, knn_idx = nbrs.kneighbors(vectors)
    knn_dist = knn_dist.astype(np.float32)
    knn_idx = knn_idx.astype(np.int32)

    # Get Gaussian sigma for local density based on neighbour matrix
    position = int(round(nb_el ** 2 * p* 0.01))
    if position >= np.prod(knn_dist.shape):
        position = -1
    dc = np.sort(knn_dist.flatten())[position]
    ordered_knn_dens = np.exp(-1 * np.square(knn_dist / dc))

    summed_okd = ordered_knn_dens.cumsum(axis=1) - 1
    rho = summed_okd[:, -1].copy()
  
    summed_okd /= int(round(p * 0.01 * nb_el))

    core_card = np.unravel_index(np.argmax(summed_okd > ordered_knn_dens, axis=1), summed_okd.shape)[1]
    return dc, knn_idx, knn_dist, rho, core_card

def get_reduce_noise(X, labels, centers):
    max_cluster_id = labels.max() + 1
    k = 0
    labels_new = labels.copy()
    centers_new = np.empty([0,0])
    for cluster_id in range(max_cluster_id):
        if np.sum(labels == cluster_id) <= X.shape[0]*0.005:
            labels_new[labels == cluster_id] = -1
        else:
            labels_new[labels == cluster_id] = k
            if k == 0:

                centers_new = centers[cluster_id]#.reshape(1,centers.shape[1])
            else:
                centers_new = np.vstack((centers_new, centers[cluster_id])) 
            k += 1
    if centers_new.size == args.n_components:
        centers_new = np.expand_dims(centers_new, axis = 0)
    if centers_new.shape[0] != 0:
        for k  in range(X.shape[0]):
            if labels_new[k] == -1:
                dist = pairwise_distances(np.expand_dims(X[k], axis = 0), centers_new)
                labels_new[k] = np.argmin(dist) 
        return labels_new, centers_new
    else:
        return labels, centers

def merge_cluster(data, dc, labels, knn_idx, knn_dist, rho, core_card):
    # center_idx
    centers_idx = []
    for i in range(labels.max()+1):
        mask_l = np.where(labels == i)[0]
        centers_idx.append(mask_l[np.argmax(rho[mask_l])])

    # Get cluster cores
    core_dict = {}
    for cl in centers_idx:
        lab = labels[cl]
        core_dict[lab] = []
        for j in knn_idx[cl, :core_card[cl]]:
            core_dict[lab].append(j)

    effective_nb_el = knn_idx.shape[1]
    border_points = {}
    for c1_idx in range(1, len(centers_idx) - 1):
        c1 = centers_idx[c1_idx]
        lab1 = labels[c1]
        mask1 = np.where(labels == lab1)[0]
        for p1 in mask1:
            for neigh in range(effective_nb_el):
                if knn_dist[p1, neigh] > dc:
                    break
                p2 = knn_idx[p1, neigh]
                lab2 = labels[p2]
                if lab2 == -1:
                    continue

                if lab1 != lab2:
                    # Order labels by density to merge in order + avoid duplicate keys
                    if rho[c1] > rho[centers_idx[lab2]]:
                        key = tuple((lab2, lab1))
                    else:
                        key = tuple((lab1, lab2))
                    if key not in border_points.keys():
                        border_points[key] = []
                    border_points[key].append(p1)
                    border_points[key].append(p2)

    # Merge based on Hellinger distance < 0.65
    cnt = 0
    for k in sorted(border_points.keys(), reverse=True):
        k0, k1 = k
        if labels[centers_idx[k0]] == labels[centers_idx[k1]]:
            continue
        border_points[k] = np.unique(border_points[k])
        mask0 = np.where(labels == labels[centers_idx[k0]])[0]
        mask1 = np.where(labels == labels[centers_idx[k1]])[0]
        try:
            kde_bord = gaussian_kde(core_card[border_points[k]])
            kde_core0 = gaussian_kde(core_card[core_dict[k0]])
            kde_core1 = gaussian_kde(core_card[core_dict[k1]])
        except Exception:
            continue
        mb = min(min(core_card[border_points[k]]), min(core_card[core_dict[k0]]), min(core_card[core_dict[k1]]))
        mc = max(max(core_card[border_points[k]]), max(core_card[core_dict[k0]]), max(core_card[core_dict[k1]]))

        support = np.linspace(mb, mc, 100)
        c2_0 = kde_core0.evaluate(support)
        c2_1 = kde_core1.evaluate(support)
        b2 = kde_bord.evaluate(support)

        c2_1 /= np.sum(c2_1)
        c2_0 /= np.sum(c2_0)
        b2 /= np.sum(b2)
        bc_0 = np.sum(np.sqrt(np.multiply(b2, c2_0)))
        bc_1 = np.sum(np.sqrt(np.multiply(b2, c2_1)))
        print(0.5*(bc_0 + bc_1))
        bc_0 = np.sqrt(1 - np.sum(np.sqrt(np.multiply(b2, c2_0))))
        bc_1 = np.sqrt(1 - np.sum(np.sqrt(np.multiply(b2, c2_1))))
        
        bc_01 = 0.5 * (bc_0 + bc_1)
        print(bc_01)
        if bc_01 < args.tol:
            labels[mask0] = labels[centers_idx[k1]]

    # Clean centers
    cur_lab = 0
    centers_idx = []
    for ulab in np.unique(labels):
        if ulab == -1:
            continue
        mask_l = np.where(labels == ulab)[0]
        centers_idx.append(mask_l[np.argmax(rho[mask_l])])
        labels[mask_l] = cur_lab
        cur_lab += 1
    return labels, data[centers_idx]


def plot(x, y, plot_id, names=None):
    viz_df = pd.DataFrame(data=x[:5000])
    viz_df['Label'] = y[:5000]
    if names is not None:
        viz_df['Label'] = viz_df['Label'].map(names)

    viz_df.to_csv(args.save_dir + '/' + args.dataset + '.csv')
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue='Label', legend='full', hue_order=sorted(viz_df['Label'].unique()),
                    palette=sns.color_palette("hls", n_colors=args.n_clusters),
                    alpha=.5,
                    data=viz_df)
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=args.n_clusters + 1, handletextpad=0.01, )

    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(args.save_dir + '/' + args.dataset +
                '-' + plot_id + '.png', dpi=300)
    plt.clf()





if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='(Not Too) Deep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', )
    parser.add_argument('gpu', default=0, type=int)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=2500, type=int)
    parser.add_argument('--num_pre_epochs', default=100, type=int)
    parser.add_argument('--num_ae_epochs', default=500, type=int)
    parser.add_argument('--num_recursive_tsne_epochs', default=50, type=int)
    parser.add_argument('--num_recursive_umap_epochs', default=100, type=int)
    parser.add_argument('--save_dir', default='results/n2d')
    parser.add_argument('--n_components', default=2, type=int)
    parser.add_argument('--umap_neighbors', default=10, type=int)
    parser.add_argument('--umap_min_dist', default="0.00", type=str)
    parser.add_argument('--umap_metric', default='euclidean', type=str)
    parser.add_argument('--cluster', default='KM', type=str)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--load_weight_emb', default=False, action='store_true')
    parser.add_argument('--p', default=4.00, type=float)
    parser.add_argument('--tol', default=0.75, type=float)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(args)

    
    from datasets import load_mnist, load_mnist_test, load_usps, load_pendigits, load_fashion, load_har, load_cifar_10, load_emnist_by, load_stl10, load_imagenet50, load_cifar_100
    from datasets import load_reuters

    label_names = None
    if args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'mnist-test':
        x, y = load_mnist_test()
    elif args.dataset == 'usps':
        x, y = load_usps()
    elif args.dataset == 'pendigits':
        x, y = load_pendigits()
    elif args.dataset == 'fashion':
        x, y, label_names = load_fashion()
    elif args.dataset == 'har':
        x, y, label_names = load_har()
    elif args.dataset == 'byclass':
        x, y = load_emnist_by()
    elif args.dataset == 'cifar10':
        x, y = load_cifar_10()
    elif args.dataset == 'cifar100':
        x, y = load_cifar_100()
    elif args.dataset == 'stl10':
        x, y = load_stl10()
    elif args.dataset == 'imagenet50':
        x, y = load_imagenet50()
    elif args.dataset == 'reuters':
        x, y = load_reuters()

   

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir + '/args.txt', 'w') as f:
        f.write("\n".join(sys.argv))
   
clusters, t_acc, t_nmi, t_ari , t_dbi, t_h= cluster_manifold_in_embedding(
        x, y, label_names)
f = open(args.save_dir + "/" + args.dataset + "-results.txt","a+")
f.write("Acc: %.5f NMI: %.5f ARI: %.5f HDB: %.5f k: %.0f h: %.5f\n" %(t_acc, t_nmi, t_ari, t_dbi, clusters.max()+1, t_h))

np.savetxt(args.save_dir + "/" + args.dataset + '-clusters.txt', clusters, fmt='%i', delimiter=',')
