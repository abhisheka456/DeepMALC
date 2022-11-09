# mnist
# python main.py mnist 0 --n_clusters=10 --batch_size=2500 --num_pre_epochs=100 --num_ae_epochs=500 --num_recursive_tsne_epochs=50 --num_recursive_umap_epochs=100 --umap_neighbors=20 --save_dir=mnist/1 --umap_min_dist=0.00 --cluster=GS --load_weight_emb


# fashion
# python \
# main.py \
# fashion \
# 0 \
# --n_clusters=10 --batch_size=2500 \
# --num_pre_epochs=100 \
# --num_ae_epochs=500 \
# --num_recursive_tsne_epochs=50 \
# --num_recursive_umap_epochs=100 \
# --umap_neighbors=20 \
# --save_dir=fashion/1 \
# --umap_min_dist=0.00 \
# --cluster=GS \
# --load_weight_emb 


# usps
python \
main.py \
usps \
0 \
--n_clusters=10 \
--batch_size=2500 \
--num_pre_epochs=100 \
--num_ae_epochs=500 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=usps/1 \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb

# pendigit
python \
main.py \
pendigits \
1 \
--n_clusters=10 \
--batch_size=2500 \
--num_pre_epochs=100 \
--num_ae_epochs=500 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=pendigits/1 \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb

# STL10
python \
main.py \
stl10 \
0 \
--n_clusters=10 \
--batch_size=2500 \
--num_pre_epochs=100 \
--num_ae_epochs=1000 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=stl10/1 \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb

# CIFAR10
python \
main.py \
cifar10 \
0 \
--n_clusters=10 \
--batch_size=2500 \
--num_pre_epochs=100 \
--num_ae_epochs=500 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=cifar10/1 \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb


# CIFAR100
python \
main.py \
cifar100 \
1 \
--n_clusters=20 \
--batch_size=2500 \
--num_pre_epochs=100 \
--num_ae_epochs=500 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=cifar100/1 \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb

# Reuters
python \
main.py \
reuters \
0 \
--n_clusters=4 \
--batch_size=1250 \
--num_pre_epochs=100 \
--num_ae_epochs=1000 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=reuters/1 \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb 
