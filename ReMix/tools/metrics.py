import os
import pdb
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy import interp
from sklearn.metrics import classification_report, confusion_matrix
from tools.cmat_metrics import CMMetrics, CMResults

rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 12

font_main_axis = {
    'weight': 'bold',
    'size': 12
}
LINE_WIDTH = 1.5

MICRO_COLOR = 'k'  # (255/255.0, 127/255.0, 0/255.0)
MACRO_COLOR = 'k'  # (255/255.0,255/255.0,51/255.0)
MICRO_LINE_STYLE = 'dashed'
MACRO_LINE_STYLE = 'solid'

CLASS_LINE_WIDTH = 2


def compute_case(slices, preds, labels, scores, savepath=None):
    case_pred = {}
    case_target = {}

    for i in range(len(slices)):
        im_p, label, pred = slices[i], int(labels[i]), int(preds[i])
        bn = os.path.basename(im_p)
        im_ind = os.path.splitext(bn)[0]
        case = im_ind.split('_')
        case = case[0] + '_' + case[1]
        if case not in case_pred:
            case_pred[case] = (pred, scores[i,:])
            case_target[case] = label
        else:
            if pred > case_pred[case][0]:
                case_pred[case] = (pred, scores[i, :])
            if pred == case_pred[case][0]:
                # check which score is higher
                s1 = case_pred[case][1]
                s2 = scores[i, :]
                if s1[pred] < s2[pred]:
                    case_pred[case] = (pred, s2)

    case_confusion_matrix = np.zeros((4, 4))
    pred_list = []
    label_list = []
    y_prob = []

    for case in case_pred:
        pred, score = case_pred[case]
        label = case_target[case]
        case_confusion_matrix[label][pred] += 1
        pred_list.append(pred)
        label_list.append(label)
        y_prob.append(score)
    
    classification_report(label_list, pred_list, digits=4)
    results_summary = dict()
    results_summary['true_labels'] = [int(x) for x in label_list]
    results_summary['pred_labels'] = [int(x) for x in pred_list]
    # print(len(y_true))
    # print(len(y_pred))
    cmat = confusion_matrix(results_summary['true_labels'], results_summary['pred_labels'])
    cmat_np_arr = np.array(cmat)
    conf_mat_eval = CMMetrics()
    cmat_results: CMResults = conf_mat_eval.compute_metrics(conf_mat=cmat_np_arr)
    cmat_results_dict = cmat_results._asdict()
    for k, v in cmat_results_dict.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        results_summary['{}'.format(k)] = v
    save_metrics(results_summary, savepath)

    # plot the ROC curves
    y_true = np.array(results_summary['true_labels'].copy())
    num_classes = np.max(y_true) + 1
    y_true = np.array(y_true, dtype=int)
    y_true_oh = np.eye(num_classes)[y_true]
    y_prob = np.array(y_prob)
    auc_macro = plot_roc(
        ground_truth=y_true_oh,
        pred_probs=y_prob,
        n_classes=num_classes,
        savepath=savepath, fname='case_level')
    
    return results_summary['overall_accuracy'], results_summary, auc_macro


class DictWriter(object):
    def __init__(self, file_name, format='csv'):
        super(DictWriter, self).__init__()
        assert format in ['csv', 'json', 'txt']

        self.file_name = '{}.{}'.format(file_name, format)
        self.format = format

    def write(self, data_dict: dict):
        if self.format == 'csv':
            import csv
            with open(self.file_name, 'w', newline="") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in data_dict.items():
                    writer.writerow([key, value])
        elif self.format == 'json':
            import json
            with open(self.file_name, 'w') as fp:
                json.dump(data_dict, fp, indent=4, sort_keys=True)
        else:
            with open(self.file_name, 'w') as txt_file:
                for key, value in data_dict.items():
                    line = '{} : {}\n'.format(key, value)
                    txt_file.write(line)


def save_metrics(metrics, save_loc):
    writer = DictWriter(file_name=save_loc, format='json')
    writer.write(metrics)


class ColorEncoder(object):
    def __init__(self):
        super(ColorEncoder, self).__init__()

    def get_colors(self, dataset_name):
        if dataset_name == 'bbwsi':
            class_colors = [
                (228/ 255.0, 26/ 255.0, 28/ 255.0),
                (55/ 255.0, 126/ 255.0, 184/ 255.0),
                (77/ 255.0, 175/ 255.0, 74/ 255.0),
                (152/ 255.0, 78/ 255.0, 163/ 255.0),
                (170/255.0, 120/255.0, 50/255.0)
            ]

            class_linestyle = ['solid', 'solid', 'solid', 'solid', 'solid']

            return class_colors, class_linestyle
        else:
            raise NotImplementedError


def plot_roc(ground_truth, pred_probs, n_classes,
             class_names=None, dataset_name='bbwsi',
             savepath=None, fname='roc_curve'):
    from sklearn.metrics import roc_curve, auc

    class_colors, class_linestyles = ColorEncoder().get_colors(dataset_name=dataset_name)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute ROC curve class-wise
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ground_truth[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # COMPUTE MICRO-AVERAGE ROC CURVE AND ROC AREA
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # COMPUTE MACRO-AVERAGE ROC CURVE AND ROC AREA

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # PLOT the curves
    micro_label = 'Micro avg. (AUC={0:0.2f})'.format(roc_auc["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=micro_label, color=MICRO_COLOR,
             linestyle=MICRO_LINE_STYLE, linewidth=LINE_WIDTH)

    macro_label = 'Macro avg. (AUC={0:0.2f})'.format(roc_auc["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=macro_label, color=MACRO_COLOR,
             linestyle=MACRO_LINE_STYLE, linewidth=LINE_WIDTH)
    class_names =['MMD', 'MIS', 'pT1a', 'pT1b']
    # pdb.set_trace()
    if class_names is not None and len(class_names) == n_classes:
        # assert len(class_names) == n_classes
        for i, c_name in enumerate(class_names):
            label = "{0} (AUC={1:0.2f})".format(c_name, roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=class_colors[i],
                     lw=CLASS_LINE_WIDTH, label=label, linestyle=class_linestyles[i])
    else:
        for i, color in zip(range(n_classes), class_colors):
            label = 'Class {0} (AUC={1:0.2f})'.format(i, roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=CLASS_LINE_WIDTH,
                     label=label, linestyle=class_linestyles[i])

    plt.plot([0, 1], [0, 1], 'tab:gray', linestyle='--', linewidth=1)
    # plt.grid(color=GRID_COLOR, linestyle=GRID_LINE_STYLE, linewidth=GRID_LINE_WIDTH)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font_main_axis)
    plt.ylabel('True Positive Rate', fontdict=font_main_axis)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(edgecolor='black', loc="best")
   
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig('{}_roc.pdf'.format(savepath), dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc["macro"]