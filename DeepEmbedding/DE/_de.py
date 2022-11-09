import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

import os
import time
import math
from tqdm import tqdm
from math import ceil
import matplotlib
import matplotlib.pyplot as plt

from ._models import *
from ._utils_de import *


# os.environ['MKL_DEBUG_CPU_TYPE'] = '5'  # 3900x


class DeepEmbedding:
    """
    ****** Deep Embedding implementation (PyTorch). ******
    Author: .......
    Version: 1.0.0
    Date: 2022/09/24

    Original Paper:

    n_components:
    The embedding dimensionality.

    n_pre_epochs:
    The number of epochs before the recursive steps.

    n_recursive_epochs:
    The number of epochs in each recursive step. (default total steps: 300+100*4)


    learning_rate:
    The learning rate of the optimization.

    tsne_perplexity (default: 30.0):
    The parameter defininig the normalization of t-SNE Pij.

    umap_n_neighbors (default: 15):
    The number of neighbors when calculating Pij in the last UMAP-like recursive step.

    min_dist:
    The minimum distance between the embedding points. Used to form Qij.

    batch_size (default: 2500):
    The batch size of the training procedure.

    rebatching_epochs:
    The number of epochs when the minibatches are fixed in the total dataset.
    """
    def __init__(self,
                 n_components=2,
                 num_ae_epochs=1000,
                 num_pre_epochs=100,
                 num_recursive_tsne_epochs=50,
                 num_recursive_umap_epochs=100,
                 learning_rate=1e-3,
                 tsne_perplexity=30.0,
                 umap_n_neighbors=15,
                 min_dist=0.001,
                 batch_size=2500,
                 rebatching_epochs=1e4,
                 save_step_models=False,
                 save_plot_results=False,
                 plot_init=False,
                 random_shuffle=True,
                 debug_mode=False,
                 save_directory='./',
                 scatter_size=1,
                 scatter_alpha=0.3,
                 dataset_name='mnist',
                 load_weight_emb=False
                 ):
        self.learn_from_exist = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_components = n_components

        self.lowest_loss = np.inf
        self.num_ae_epochs = num_ae_epochs
        self.num_pre_epochs = num_pre_epochs
        self.num_recursive_tsne_epochs = num_recursive_tsne_epochs
        self.num_recursive_umap_epochs = num_recursive_umap_epochs
        self.num_epochs = self.num_ae_epochs  + self.num_pre_epochs + 3*self.num_recursive_tsne_epochs + self.num_recursive_umap_epochs
        self.batch_size = batch_size

        self.perplexity = tsne_perplexity
        self.umap_knn = umap_n_neighbors
        self.min_dist = min_dist
        self.a, self.b = find_ab_params(1, self.min_dist)

        self.lr = learning_rate
        self.dataset_name = dataset_name
        self.load_weight_emb = load_weight_emb

        self.rolling_num = rebatching_epochs
        self.plotting_num = 5

        self.P = []
        self.data = 0
        self.num_batch = 0
        self.embedding = 0
        self.save_steps = save_step_models
        self.plot_results = save_plot_results
        self.plot_init = plot_init
        self.directory = save_directory

        self.loss_plot_train = []
        self.loss_plot_val = []



        # placeholder
        self.net = 0
        self.net_optim = 0
        self.data_dim = 0
        self.optimizer = 0
        self.random_shuffle = random_shuffle

        # Debug mode:
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.labels = 0
            self.colors = ['darkorange', 'deepskyblue', 'gold', 'lime', 'k', 'darkviolet', 'peru', 'olive',
                           'midnightblue',
                           'palevioletred']
            self.cmap = matplotlib.colors.ListedColormap(self.colors[::-1])

        self.scatter_size = scatter_size
        self.scatter_alpha = scatter_alpha

    def calculate_p_matrix(self, step):
        if self.random_shuffle:
            ran_num = np.random.randint(2 ** 16 - 1)
            np.random.seed(ran_num)
            np.random.shuffle(self.data)
        return self._calculate_p_matrix(step)  # fill in 'pre', 're1', 're2', 're3', 're_umap'

    def _calculate_p_matrix(self, step):
        print('building P matrix...')
        self.P = []
        self.net.eval()
        with torch.no_grad():
            if step == 'ae':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    self.P.append(inputs)
                    print('[DE] autoencoder step 1 data shape: ', inputs.shape)
            if step == 'pre':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
                    low_dim_data, _, _, _, _, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu()).reshape(low_dim_data.shape[0], -1)
                    print('[DE] recursive step 1 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DE] P length: ', len(self.P))
            if step == 're1':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)

                    _, _, low_dim_data, _, _, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu()).reshape(inputs.shape[0], -1)
                    print('[DE] recursive step 1 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DE] P length: ', len(self.P))
            if step == 're2':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
                    _, _, _, low_dim_data, _, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu()).reshape(inputs.shape[0], -1)
                    print('[DE] recursive step 2 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DE] P length: ', len(self.P))
            if step == 're3':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
                    _, _, _, _, low_dim_data, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu()).reshape(inputs.shape[0], -1)
                    print('[DE] recursive step 3 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='tsne')
                    self.P.append(P1)
                    print('[DE] P length: ', len(self.P))
            if step == 're_umap':
                for i in range(self.num_batch):
                    inputs = self.data[i * self.batch_size:(i + 1) * self.batch_size].astype('float32')
                    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(self.device)
                    _, _, _, _, low_dim_data, _ = self.net(inputs)
                    low_dim_data = np.array(low_dim_data.cpu()).reshape(inputs.shape[0], -1)
                    print('[DE] recursive step 3 data shape: ', low_dim_data.shape)
                    P1 = conditional_probability_p(self.perplexity, self.umap_knn, low_dim_data, p_type='umap')
                    self.P.append(P1)
                    print('[DE] P length: ', len(self.P))

        return self.P

    def _train(self, epoch, step):
        # print('\nEpoch: %d' % epoch)
        self.net.train()  # train mode
        loss_all = 0
        with tqdm(total=self.data.shape[0], desc=f'[DE] Training.. Epoch {epoch + 1}/{self.num_epochs}', unit='img', colour='blue') as pbar:
            for i in range(self.num_batch):
                # Load the packed data:
                tar = self.P[i]
                inputs, targets = torch.from_numpy(self.data[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor).to(self.device),\
                                  torch.from_numpy(tar).type(torch.FloatTensor).to(self.device)

                inputs = inputs.reshape(inputs.shape[0], -1)
                self.optimizer.zero_grad()
                if step == 'ae':
                    self.net.fc0.weight.requires_grad = True
                    self.net.fc1.weight.requires_grad = True
                    self.net.fc2.weight.requires_grad = True
                    self.net.fc5.weight.requires_grad = True
                    self.net.fc6.weight.requires_grad = True
                    self.net.fc7.weight.requires_grad = True
                    self.net.fc8.weight.requires_grad = True
                    self.net.fc9.weight.requires_grad = True

                    self.net.fc10.weight.requires_grad = False
                    self.net.fc11.weight.requires_grad = False
                    self.net.fc12.weight.requires_grad = False
                    self.net.fc15.weight.requires_grad = False
                    self.net.fc16.weight.requires_grad = False
                    self.net.fc17.weight.requires_grad = False
                    self.net.fc18.weight.requires_grad = False

                    _, outputs, _, _, _, _ = self.net(inputs)
                else:
                    self.net.fc0.weight.requires_grad = False
                    self.net.fc1.weight.requires_grad = False
                    self.net.fc2.weight.requires_grad = False
                    self.net.fc5.weight.requires_grad = False
                    self.net.fc6.weight.requires_grad = False
                    self.net.fc7.weight.requires_grad = False
                    self.net.fc8.weight.requires_grad = False
                    self.net.fc9.weight.requires_grad = False

                    self.net.fc10.weight.requires_grad = True
                    self.net.fc11.weight.requires_grad = True
                    self.net.fc12.weight.requires_grad = True
                    self.net.fc15.weight.requires_grad = True
                    self.net.fc16.weight.requires_grad = True
                    self.net.fc17.weight.requires_grad = True
                    self.net.fc18.weight.requires_grad = True

                    _, _, _, _, _, outputs = self.net(inputs)

                loss = loss_function(targets, outputs, self.a, self.b, type=step)
                # print(loss)
                if str(loss.item()) == str(np.nan):
                    print('[DE] detect nan in loss function, skip this iter')
                    continue
                loss.backward()
                loss_all += loss.item()
                # losses[divide_number*i+j] = loss
                self.optimizer.step()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(targets.shape[0])
        return loss_all / self.num_batch


    def _validation(self, epoch, step):
        self.net.eval()  # train mode
        loss_all = 0
        with torch.no_grad():
            with tqdm(total=self.data.shape[0], desc=f'[DE] Validating.. Epoch {epoch + 1}/{self.num_epochs}', unit='img',
                      colour='green') as pbar:
                for i in range(self.num_batch):
                    # Load the packed data:
                    tar = self.P[i]
                    inputs, targets = torch.from_numpy(self.data[i*self.batch_size:(i+1)*self.batch_size]).type(torch.FloatTensor).to(self.device), \
                                      torch.from_numpy(tar).type(torch.FloatTensor).to(self.device)
                    inputs = inputs.reshape(inputs.shape[0], -1)
                    if step == 'ae':
                        self.net.fc0.weight.requires_grad = True
                        self.net.fc1.weight.requires_grad = True
                        self.net.fc2.weight.requires_grad = True
                        self.net.fc5.weight.requires_grad = True
                        self.net.fc6.weight.requires_grad = True
                        self.net.fc7.weight.requires_grad = True
                        self.net.fc8.weight.requires_grad = True
                        self.net.fc9.weight.requires_grad = True

                        self.net.fc10.weight.requires_grad = False
                        self.net.fc11.weight.requires_grad = False
                        self.net.fc12.weight.requires_grad = False
                        self.net.fc15.weight.requires_grad = False
                        self.net.fc16.weight.requires_grad = False
                        self.net.fc17.weight.requires_grad = False
                        self.net.fc18.weight.requires_grad = False

                        _, outputs, _, _, _, _ = self.net(inputs)
                    else:
                        self.net.fc0.weight.requires_grad = False
                        self.net.fc1.weight.requires_grad = False
                        self.net.fc2.weight.requires_grad = False
                        self.net.fc5.weight.requires_grad = False
                        self.net.fc6.weight.requires_grad = False
                        self.net.fc7.weight.requires_grad = False
                        self.net.fc8.weight.requires_grad = False
                        self.net.fc9.weight.requires_grad = False

                        self.net.fc10.weight.requires_grad = True
                        self.net.fc11.weight.requires_grad = True
                        self.net.fc12.weight.requires_grad = True
                        self.net.fc15.weight.requires_grad = True
                        self.net.fc16.weight.requires_grad = True
                        self.net.fc17.weight.requires_grad = True
                        self.net.fc18.weight.requires_grad = True


                        _, _, _, _, _, outputs = self.net(inputs)
                    # loss = kl_divergence_bayes(outputs, targets, 1, knn_bayes)
                    loss = loss_function(targets, outputs, self.a, self.b, type=step)
                    loss_all += loss
                    # loss_aver = loss_all / (i+1)
                    pbar.set_postfix(**{'loss (batch)': loss_all.item() / self.num_batch})
                    pbar.update(targets.shape[0])
        # Save checkpoint.
        if float(loss_all / self.num_batch) < self.lowest_loss:
            print('[DE] Best accuracy, saving the weights...')
            self.net_optim = self.net
            lowest_loss = float(loss_all / self.num_batch)
        return loss_all / self.num_batch


    def plot(self, epoch, step):
        self.embedding = self._plot(epoch, step)

    def _plot(self, epoch, step, fig_size='normal'):
        self.net.eval()
        with torch.no_grad():
            for i in range(self.num_batch):


                inputs = torch.from_numpy(self.data[i * self.batch_size:(i + 1) * self.batch_size])\
                        .type(torch.FloatTensor).to(self.device)

                
                if i == 0:
                    _, _, _, _, _, Y = self.net(inputs)
                    Y = Y.cpu()
                else:
                    _, _, _, _, _, y_test = self.net(inputs)
                    Y = np.concatenate((Y, y_test.cpu()), axis=0)
        torch.cuda.empty_cache()
        if self.plot_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            if fig_size == 'fixed':
                plt.xlim([-500, 500])
                plt.ylim([-500, 500])
            if self.debug_mode:
                scatter = ax.scatter(Y[:, 0], Y[:, 1], s=self.scatter_size, cmap=self.cmap, c=self.labels, alpha=self.scatter_alpha)
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
                if step == 'ae':
                    ax.text(0.03, 0.03, 'Stage 1\nAutoencoder',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            fontsize=30,
                            transform=ax.transAxes)
                if step == 'pre':
                    ax.text(0.03, 0.03, 'Stage 1\nNo recursion',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            fontsize=30,
                            transform=ax.transAxes)
                elif step == 're1':
                    ax.text(0.03, 0.03, 'Stage 1\nRecursion 1',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            fontsize=30,
                            transform=ax.transAxes)
                elif step == 're2':
                    ax.text(0.03, 0.03, 'Stage 1\nRecursion 2',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            fontsize=30,
                            transform=ax.transAxes)
                elif step == 're3':
                    ax.text(0.03, 0.03, 'Stage 1\nRecursion 3',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            fontsize=30,
                            transform=ax.transAxes)
                elif step == 're_umap':
                    ax.text(0.03, 0.03, 'Stage 2\nRecursion UMAP',
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            fontsize=30,
                            transform=ax.transAxes)
            else:
                scatter = ax.scatter(Y[:, 0], Y[:, 1], s=self.scatter_size, c='darkorange', alpha=self.scatter_alpha)
            # plt.title('Epoch = %d, Loss = %f' % (epoch, self.loss_score_train))
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.tight_layout()
            plt.axis('equal')
            plt.savefig(self.directory + "DE_labeled_%s_%s.png" % (time.asctime(time.localtime(time.time())), step))
            plt.close()
        if step == 're_umap':
            return Y
        else:
            return 0
    

    def _fit_embedding(self):
        start_time = time.time()
        print('[DE] start------->  time: ', time.ctime(time.time()))
        recursive_step = 'ae'
        for epoch in range(self.num_epochs):

            if epoch == 0:
                self.P = self.calculate_p_matrix(recursive_step)

            if self.plot_results and self.plot_init and epoch < 5:  # plot embedding results in the first steps
                self.plot(epoch, recursive_step)
                
            if epoch % self.plotting_num == 0 and epoch != 0 and self.plot_results:
                self.plot(epoch, recursive_step)

            if epoch % self.rolling_num == 0 and epoch != 0 and epoch not in \
                    np.int16(self.num_pre_epochs + np.int16([0, 1, 2, 3])*self.num_recursive_tsne_epochs):
                self.P = self.calculate_p_matrix(recursive_step)
            
            if epoch == self.num_ae_epochs:  # 300
                if self.save_steps:
                    # save the model:
                    state = {
                        'net': self.net.state_dict(),
                        'loss': self.loss_score_train,
                        'epoch': epoch,
                    }
                    if not os.path.isdir(self.directory+'DE_model_checkpoint'):
                        os.mkdir(self.directory+'DE_model_checkpoint')
                    torch.save(state, self.directory+'DE_model_checkpoint/DRE_{}.pth'.format(recursive_step))
                if self.plot_results:
                    self.plot(epoch, recursive_step)

                recursive_step = 'pre'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_ae_epochs + self.num_pre_epochs:  # 300
                if self.save_steps:
                    # save the model:
                    state = {
                        'net': self.net.state_dict(),
                        'loss': self.loss_score_train,
                        'epoch': epoch,
                    }
                    if not os.path.isdir(self.directory+'DE_model_checkpoint'):
                        os.mkdir(self.directory+'DE_model_checkpoint')
                    torch.save(state, self.directory+'DE_model_checkpoint/DRE_{}.pth'.format(recursive_step))
                if self.plot_results:
                    self.plot(epoch, recursive_step)

                recursive_step = 're1'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_ae_epochs + self.num_pre_epochs + self.num_recursive_tsne_epochs:  # 400
                if self.save_steps:
                    # save the model:
                    state = {
                        'net': self.net.state_dict(),
                        'loss': self.loss_score_train,
                        'epoch': epoch,
                    }
                    torch.save(state, self.directory+'DE_model_checkpoint/DRE_{}.pth'.format(recursive_step))

                if self.plot_results:
                    self.plot(epoch, recursive_step)

                recursive_step = 're2'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_ae_epochs + self.num_pre_epochs + 2*self.num_recursive_tsne_epochs:  # 500
                if self.save_steps:
                    # save the model:
                    state = {
                        'net': self.net.state_dict(),
                        'loss': self.loss_score_train,
                        'epoch': epoch,
                    }
                    torch.save(state, self.directory+'DE_model_checkpoint/DRE_{}.pth'.format(recursive_step))
                
                if self.plot_results:
                    self.plot(epoch, recursive_step)

                recursive_step = 're3'
                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            if epoch == self.num_ae_epochs + self.num_pre_epochs + 3*self.num_recursive_tsne_epochs:  # 600
                if self.save_steps:
                    # save the model:
                    state = {
                        'net': self.net.state_dict(),
                        'loss': self.loss_score_train,
                        'epoch': epoch,
                    }
                    torch.save(state, self.directory+'DE_model_checkpoint/DRE_{}.pth'.format(recursive_step))

                if self.plot_results:
                    self.plot(epoch, recursive_step)

                recursive_step = 're_umap'

                # adjust the learning tate for convolutional DRE:
                self.lr = 1e-4
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-7)

                # calculate the new P matrix:
                self.P = self.calculate_p_matrix(recursive_step)

            # <==================== train ====================>

            self.loss_score_train = self._train(epoch, recursive_step)


            # <==================== validate ====================>

            if (epoch + 1) % 10 == 0:
                self.loss_score_val = self._validation(epoch, recursive_step).cpu()

                self.loss_plot_val.append(self.loss_score_val)
            # scheduler.step()
            self.loss_plot_train.append(self.loss_score_train)

        end_time = time.time()
        duration = end_time - start_time
        print('[DE] training time: ', duration)
        print('[DE] ------->complete.  time: ', time.ctime(time.time()))

    

        self.plot(epoch, recursive_step)

    def _fit(self, x):
        self.data = x

        self.data_dim = self.data.shape[1]

        print('[DE] Building model...')

        self.net = DEC(self.data_dim, self.n_components)
        
        


        self.net = self.net.to(self.device)
        # if self.device == 'cuda':
        #     self.net = torch.nn.DataParallel(self.net)
        #     cudnn.benchmark = True
        self.net_optim = 0
        
        PATH = self.directory+'weights/'+self.dataset_name+'_embedd.zip'
        
        
        if self.load_weight_emb:
            print('[DE] Model loaded')
            self.net.load_state_dict(torch.load(PATH))
            _, _, _, _, _, self.embedding = self.net(torch.from_numpy(self.data).type(torch.FloatTensor).to(self.device))
            self.embedding = self.embedding.detach().cpu().numpy()
        else:
            print('[DE] Model training is started')
            # Loss function and optimization method:

            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-7)
            # self.optimizer = optim.SGD(net.parameters(), lr=1e-4,
            #                       momentum=0.9, weight_decay=5e-4)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1500)

            self.num_batch = ceil(self.data.shape[0] / self.batch_size)

            start_time = time.time()
            self._fit_embedding()
            end_time = time.time()
            torch.save(self.net.state_dict(), PATH)

            print('fitting time: {}s'.format(end_time - start_time))

        return self.embedding

    def fit_transform(self, x):
        _embedding = self._fit(x)
        self.embedding = _embedding
        return self.embedding

    def fit(self, x):
        self.fit_transform(x)
        return self

    def save_model(self, save_mode='last_epoch', save_dir='./', model_name='DE_manually_save'):
        if self.net == 0:
            raise TypeError('[DE] fit the model first')
        if save_mode == 'last_epoch':
            state = {
                'net': self.net.state_dict(),
                'loss': self.loss_score_val.item(),
                'epoch': self.num_epochs,
            }
        elif save_mode == 'lowest_loss':
            state = {
                'net': self.net_optim.state_dict(),
                'loss': self.lowest_loss.item(),
                'epoch': self.num_epochs,
            }
        else:
            raise TypeError('[DE] save_mode invalid')
        if not os.path.isdir(save_dir+'DE_model_checkpoint'):
            os.mkdir(save_dir+'DE_model_checkpoint')
        # Remember to rename the saved model corresponding to the hyper-parameters:
        torch.save(state, save_dir+'DE_model_checkpoint/{}.pth'.format(model_name))



