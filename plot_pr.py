import matplotlib
matplotlib.use('svg')
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt


plt.clf()
filename = ['CNN+ATT', 'Hoffmann', 'MIMLRE', 'Mintz', 'PCNN+ATT']
color = ['red', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
for i in range(len(filename)):
    precision = np.load('./data/' + filename[i] + '_precision.npy')
    recall = np.load('./data/' + filename[i] + '_recall.npy')
    plt.plot(recall, precision, color=color[i], lw=2, label=filename[i])

# ATTENTION: put the model iters you want to plot into the list
model_iter = [10900]
for one_iter in model_iter:
    y_true = np.load('./data/allans.npy')
    # careful with the predicted probability output file path and NAME!
    y_scores = np.load('./out/allprob_iter_' + str(one_iter) + '.npy')

    precision, recall, threshold = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.plot(recall[:], precision[:], lw=2, color='navy', label='BGRU+2ATT')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.3, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig('iter_' + str(one_iter) + '.svg')
