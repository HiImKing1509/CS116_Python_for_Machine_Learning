import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

is_loaded = False
is_train = False

def plot_kfold(metric_arr, fold, color):
    # set width of bar
    barWidth = 0.5
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    metric = metric_arr
    
    # Set position of bar on X axis
    br1 = np.arange(len(metric))
    
    # Make the plot
    plt.bar(br1, metric, color = color, width = barWidth, edgecolor ='grey', label ='metric')
    
    # Adding Xticks
    plt.xlabel('Fold', fontweight ='bold', fontsize = 16)
    plt.ylabel('Score', fontweight ='bold', fontsize = 16)
    plt.xticks(
                [r for r in range(len(metric))],
                fold
        )
        
    plt.legend()
    plt.close()