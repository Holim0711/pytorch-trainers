from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(true, pred, class_names):
    cm = confusion_matrix(true, pred, labels=list(range(len(class_names))))

    figsize = (0.8 * len(class_names) + 4.8, 0.8 * len(class_names))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.matshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(list(range(len(class_names))))
    ax.set_yticks(list(range(len(class_names))))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    ax.margins(0.5)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return fig


class Metric():

    def __init__(self):
        self.class_names = list(map(str, range(10)))
        self.pred, self.true = [], []

    def update(self, pred, true):
        self.pred += pred.argmax(dim=1).tolist()
        self.true += true.tolist()

    def output(self):
        accuracy = accuracy_score(self.true, self.pred)
        figure = plot_confusion_matrix(self.true, self.pred, self.class_names)
        self.pred, self.true = [], []
        return accuracy, figure
