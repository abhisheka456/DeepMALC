# mnist
python \
main_gs_mod.py \
mnist \
0 \
--n_clusters=10 \
--batch_size=2500 \
--num_pre_epochs=100 \
--num_ae_epochs=500 \
--num_recursive_tsne_epochs=50 \
--num_recursive_umap_epochs=100 \
--umap_neighbors=20 \
--save_dir=mnist/gs_ab \
--umap_min_dist=0.00 \
--cluster=GS \
--load_weight_emb \
--p=7.0 \

