from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from models import Model
import torch
from tqdm import tqdm
import os
from configs import cfg

def visualize(model, test_loader):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    test_embeds_arr = []
    test_labels_arr = []
    model.eval()

    for (w2v2, mfcc, mel, lld, labels) in tqdm(test_loader):
        with torch.no_grad():

            embeds, _ = model(w2v2.to(cfg.device), mfcc.to(cfg.device), mel.to(cfg.device), lld.to(cfg.device))
            
            
            test_embeds_arr.extend(embeds.cpu().detach().numpy())
            test_labels_arr.extend(labels)

    X = np.array(test_embeds_arr).squeeze()
    y = np.array(test_labels_arr)

    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    print('Size of the dataframe: {}'.format(df.shape))

    df_subset = df.copy()

    X = np.array(test_embeds_arr).squeeze()

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(X)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['TSNE-x'] = tsne_results[:,0]
    df_subset['TSNE-y'] = tsne_results[:,1]
    df_subset['y'] = y

    plt.figure(figsize=(8,5))

    ax = sns.scatterplot(
        x="TSNE-x", y="TSNE-y",
        hue="y",
        palette=sns.color_palette("hls", cfg.n_classes),
        data=df_subset,
        legend="full",
    )

    handles, labels  =  ax.get_legend_handles_labels()

    plt.legend(handles = handles, labels = sorted(os.listdir(cfg.train_path)))

    df_subset['x'] = tsne_results[:,0]
    df_subset['y'] = tsne_results[:,1]
    df_subset['hue'] = y

    plt.figure(figsize=(8,5))
        
    ax = sns.scatterplot(
        x="x", y="y",
        hue="hue",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        
    )

    for line in range(0,df.shape[0]):
        plt.text(df_subset.x[line]+0.2, df_subset.y[line], str(line), horizontalalignment='left', fontsize=4, color='black', weight='semibold')
        
    handles, labels  =  ax.get_legend_handles_labels()

    plt.legend(handles = handles, labels = sorted(os.listdir(cfg.train_path)))

    temp = []
    class_reps = []

    for i in range(cfg.n_classes):
        for j in range(len(test_labels_arr)):
            if test_labels_arr[j] == i:
                temp.append(test_embeds_arr[j])

        class_reps.append(np.mean(np.array(temp), axis = 0))
        temp = []

    class_reps = np.array(class_reps)

    from sklearn.metrics.pairwise import cosine_similarity
    labels = sorted(os.listdir(cfg.train_path))

    figure = plt.figure()
    axes = figure.add_subplot(111)

    sim_matrix = cosine_similarity(class_reps, class_reps)
    print(sim_matrix)
                
    caxes = axes.matshow(sim_matrix, interpolation ='nearest')
    figure.colorbar(caxes)

    for (i, j), z in np.ndenumerate(sim_matrix):
        axes.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    axes.set_xticklabels(['']+labels)
    axes.set_yticklabels(['']+labels)
    plt.xticks(rotation=90)
    plt.show()

    (np.sum(sim_matrix, axis = 1) -1 )/ (cfg.n_classes-1)
    sample_arr = []

    for i in df_subset.index:
        sample = np.array([df_subset['x'][i], df_subset['y'][i]])
        sample_arr.append(sample)
        
    sample_arr = np.array(sample_arr)  


    temp = []
    class_reps = []

    for i in range(cfg.n_classes):
        for j in range(len(test_labels_arr)):
            if test_labels_arr[j] == i:
                temp.append(sample_arr[j])

        class_reps.append(np.mean(np.array(temp), axis = 0))
        temp = []

    class_reps = np.array(class_reps)

    from sklearn.metrics.pairwise import cosine_similarity
    labels = sorted(os.listdir(cfg.train_path))

    figure = plt.figure()
    axes = figure.add_subplot(111)

    sim_matrix = cosine_similarity(class_reps, class_reps)
    print(sim_matrix)
                
    caxes = axes.matshow(sim_matrix, interpolation ='nearest')
    figure.colorbar(caxes)

    for (i, j), z in np.ndenumerate(sim_matrix):
        axes.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')

    axes.set_xticklabels(['']+labels)
    axes.set_yticklabels(['']+labels)
    plt.xticks(rotation=90)

    plt.show()