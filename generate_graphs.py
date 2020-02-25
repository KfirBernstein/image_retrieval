import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import collections

med_net_to_batches = {'AlexNet':[30,200],
                      'VGG19':[30],
                      'VGG16': [30,40],
                      'ResNet18':[30,150],
                      'GoogLeNet': [30,75]}
med_net_to_colors = {'AlexNet':'red',
                     'VGG19':'green',
                     'VGG16': 'black',
                     'ResNet18':'blue',
                     'GoogLeNet': 'orange'}
image_indexes = [5,1005,3005,5005,7005,9005]

labels = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
shared_batch = 30

if __name__ == '__main__':
    indexes = range(1000)
    if not os.path.isdir('graphs'):
        os.mkdir('graphs')
    ##P@k Shared Batch ####
    for med_net in med_net_to_batches:
        prec = np.load('model_'+med_net+'_'+str(shared_batch)+'/precision_at_k.npy')[:1000]
        plt.plot(indexes,prec,med_net_to_colors[med_net])
    plt.title('Prec@k Batch ='+str(shared_batch))
    plt.ylabel('Precision')
    plt.xlabel('k')
    plt.savefig('graphs/shared_batch_prec_at_k.png')
    plt.clf()
    ###P@k Max Batch ####
    for med_net in med_net_to_batches:
        max_batch = max(med_net_to_batches[med_net])
        prec = np.load('model_'+med_net+'_'+str(max_batch)+'/precision_at_k.npy')[:1000]
        plt.plot(indexes,prec,med_net_to_colors[med_net])
    plt.title('Prec@k Max Batch')
    plt.ylabel('Precision')
    plt.xlabel('k')
    legend = plt.legend(list(med_net_to_batches.keys()),loc = 'best')
    fig = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.legend().set_visible(False)
    fig.savefig("graphs/legend.png", dpi="figure", bbox_inches=bbox)
    plt.savefig('graphs/max_batch_prec_at_k.png')
    plt.clf()
    ###MAP ####
    med_net_map = {}
    for med_net in med_net_to_batches:
        map = float("%.4f" %np.mean(np.load('model_'+med_net+'_'+str(shared_batch)+'/AP_per_query.npy')))
        med_net_map[med_net] = [map]
    for med_net in med_net_to_batches:
        max_batch = max(med_net_to_batches[med_net])
        map = float("%.4f" %np.mean(np.load('model_'+med_net+'_'+str(max_batch)+'/AP_per_query.npy')))
        med_net_map[med_net].append(map)
    for med_net in med_net_to_batches:
        imp = str("%.2f" %round(((med_net_map[med_net][1]/med_net_map[med_net][0])-1)*100,2)) + '%'
        med_net_map[med_net].append(imp)
    df = pd.DataFrame(data=med_net_map,index = [' Batch = '+str(shared_batch),' Max Batch',' Improvment'])
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    tab = pd.plotting.table(ax, df, loc='upper right',cellLoc  = 'center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    plt.savefig('graphs/map.png')
    plt.clf()
    ###Test Acc  Shared Batch ####
    indexes = range(50)
    for med_net in med_net_to_batches:
        f = open('model_'+med_net+'_'+str(shared_batch)+'/test_acc.txt', "r")
        l = [float(line.split()[0]) for line in f]
        plt.plot(indexes,l,med_net_to_colors[med_net])
    plt.title('Test Accuracy Batch ='+str(shared_batch))
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylim((70,100))
    plt.yticks(np.arange(70, 100, step=5),np.arange(70, 100, step=5))
    plt.savefig('graphs/shared_batch_test_acc.png')
    plt.clf()
    ###Test Acc  Max Batch ####
    indexes = range(50)
    for med_net in med_net_to_batches:
        max_batch = max(med_net_to_batches[med_net])
        f = open('model_'+med_net+'_'+str(max_batch)+'/test_acc.txt', "r")
        l = [float(line.split()[0]) for line in f]
        plt.plot(indexes,l,med_net_to_colors[med_net])
    plt.title('Test Accuracy Max Batch')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylim((70,100))
    plt.yticks(np.arange(70, 100, step=5),np.arange(70, 100, step=5))
    plt.savefig('graphs/max_batch_test_acc.png')
    plt.clf()
    med_net_map = {}
    for med_net in med_net_to_batches:
        map = float("%.4f" %np.mean(np.load('model_'+med_net+'_'+str(shared_batch)+'/AP_per_query.npy')))
        f = open('model_'+med_net+'_'+str(shared_batch)+'/test_acc.txt', "r")
        l = [float(line.split()[0]) for line in f]
        acc = round(float(l[-1])/100,4)
        med_net_map[med_net] = [acc,map]
    for med_net in med_net_to_batches:
        imp = str("%.2f" %round(((med_net_map[med_net][1]/med_net_map[med_net][0])-1)*100,2)) + '%'
        med_net_map[med_net].append(imp)
    df = pd.DataFrame(data=med_net_map,index = [' Test Accuracy',' Batch = '+str(shared_batch),' Improvment'])
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    tab = pd.plotting.table(ax, df, loc='upper right',cellLoc  = 'center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.5)
    plt.savefig('graphs/shared_batch_map_test_acc.png')
    plt.clf()
    ##Test Acc to MAP max batch ####
    med_net_map = {}
    for med_net in med_net_to_batches:
        max_batch = max(med_net_to_batches[med_net])
        map = float("%.4f" %np.mean(np.load('model_'+med_net+'_'+str(max_batch)+'/AP_per_query.npy')))
        f = open('model_'+med_net+'_'+str(max_batch)+'/test_acc.txt', "r")
        l = [float(line.split()[0]) for line in f]
        acc = round(float(l[-1])/100,4)
        med_net_map[med_net] = [acc,map]
    for med_net in med_net_to_batches:
        imp = str("%.2f" %round(((med_net_map[med_net][1]/med_net_map[med_net][0])-1)*100,2)) + '%'
        med_net_map[med_net].append(imp)
    df = pd.DataFrame(data=med_net_map,index = [' Test Accuracy',' Max Batch',' Improvment'])
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    tab = pd.plotting.table(ax, df, loc='upper right',cellLoc  = 'center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.5)
    plt.savefig('graphs/max_batch_map_test_acc.png')
    plt.clf()
    #labels histogram ####
    for med_net in med_net_to_batches:
        for i in image_indexes:
            labels_for_net = np.load('model_'+med_net+'_'+str(shared_batch)+'/retrieval/'+str(i)+'/labels.npy')
            labels_for_net = [labels[int(idx)]for idx in labels_for_net]
            label_to_count = collections.Counter(labels_for_net)
            print (med_net,i)
            print (label_to_count)
            plt.title('Labels Histogram (top 1000)')
            plt.ylabel('count')
            plt.xlabel('class')
            keys = [labels[idx] for idx in labels]
            values = []
            for key in keys:
                if key in label_to_count:
                    values.append(label_to_count[key])
                else:
                    values.append(0)
            bars = plt.bar(keys,values)
            for bar in bars:
                yval = bar.get_height()
                newVal = str(yval)
                newVal = (4-len(newVal))*' '+newVal
                plt.text(bar.get_x(), yval + .25, newVal,fontweight='bold')
            plt.xticks(fontsize=6.5)
            plt.yticks(fontsize=6.5)
            plt.savefig('model_'+med_net+'_'+str(shared_batch)+'/retrieval/'+str(i)+'/histogram.png')
            plt.clf()