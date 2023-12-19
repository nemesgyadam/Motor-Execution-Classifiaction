import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns


model_paths = sorted(glob('checkpoints/*'))
sns.set(style="whitegrid")
models_ignored = ['EEGNetv4']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
labels = []  # initialize an empty list to keep track of labels for the legend
legend_set = set()
color_mapping = {}

for model_path in model_paths:
    model_dir = model_path[model_path.rfind('/') + 1:]
    model_cls = model_dir.split('_')[2]
    train_obj = model_dir.split('_')[1]
    if model_cls in models_ignored:
        continue

    metrics = pd.read_csv(f'{model_path}/training_logs/version_0/metrics.csv')

    val_rows = ~pd.isna(metrics['val_acc'])
    step = metrics['step'].loc[val_rows].to_numpy()
    val_loss = metrics['val_loss'].loc[val_rows].to_numpy()[step < 100000]  # truncate
    val_acc = metrics['val_acc'].loc[val_rows].to_numpy()[step < 100000]
    step = step[step < 100000]

    # Generate a unique color for each model
    if model_cls not in color_mapping:
        color_mapping[model_cls] = next(ax2._get_lines.prop_cycler)['color']

    color = color_mapping[model_cls]
    # color = next(ax2._get_lines.prop_cycler)['color']

    # plot
    val_acc_percent = val_acc * 100
    if train_obj == 'single':
        sns.lineplot(x=step, y=val_acc_percent, ax=ax1, color=color, label=model_cls, marker='o', alpha=0.75, markersize=5.2)
        # ax1.set_title('Validation Accuracy')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Validation Accuracy (%)')
        ax1.legend().remove()

    else:  # multi
        sns.lineplot(x=step, y=val_acc_percent, ax=ax2, color=color, label=model_cls, marker='o', alpha=0.75, markersize=5.2)
        # ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Validation Accuracy (%)')
        ax2.legend().remove()

    if model_cls not in legend_set:
        labels.append(model_cls)
        legend_set.add(model_cls)

plt.tight_layout()

ax1.text(-0.05, 1.08, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
ax2.text(-0.05, 1.08, 'B', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

plt.subplots_adjust(top=0.92)
handles, _ = ax1.get_legend_handles_labels()
plt.figlegend(handles, labels, loc='lower right', ncol=1, frameon=True, bbox_to_anchor=(0.975, 0.11), borderpad=.6)
plt.savefig('tmp/dl_model_accs.png')
plt.savefig('tmp/dl_model_accs.pdf')
plt.show()

print('a')
