import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read excel file
df = pd.read_excel('Swap-In compare with re-compute.xlsx', header=1)

# import data
x = df['token number'].map(lambda a: pd.to_numeric(a[:-1]))
y1_1 = df['recompute latency']
y1_2 = df['swap in latency']
y2_1 = df['recompute latency.1']
y2_2 = df['swap in latency.1']
y3_1 = df['recompute latency.2']
y3_2 = df['swap in latency.2']
y4_1 = df['recompute latency.3']
y4_2 = df['swap in latency.3']

def plot_bar(x, y1, y2, name, yticks, title):
    plt.figure(figsize=(6, 4))
    # plot
    width = 0.15
    plt.bar(x-width, y1, width*2, label='recompute latency')
    plt.bar(x+width, y2, width*2, label='reload latency')

    # add title and label
    plt.title(title)
    plt.xlabel('Token Number (k)', fontdict={'size': 14})
    plt.ylabel('Latency (s)', fontdict={'size': 14})

    # show axies
    plt.xlim(0, 32)
    plt.xticks(np.arange(0, 33, 4))
    plt.yticks(yticks)

    # legend
    plt.legend(loc='upper left')

    # show figure
    # plt.show()
    plt.tight_layout()
    plt.savefig(name)


# plot_bar(x, y1_1, y1_2, 'latency_compare_OPT-6.7B.jpg', np.arange(0, 5, 1), 'OPT-6.7B')
# plot_bar(x, y2_1, y2_2, 'latency_compare_OPT-13B.jpg', np.arange(0, 8, 2), 'OPT-13B')
# plot_bar(x, y3_1, y3_2, 'latency_compare_OPT-30B.jpg', np.arange(0, 5, 1), 'OPT-30B')
# plot_bar(x, y4_1, y4_2, 'latency_compare_Llama-2-7b-longlora.jpg', np.arange(0, 5, 1), 'Llama-2-7b-longlora')

def plot_scartter(x, y1, y2, name, yticks, title):
    plt.figure(figsize=(6, 4))
    # plot
    # dot
    plt.plot(x, y1, 'o-', label='recompute latency')
    plt.plot(x, y2, 'o-', label='reload latency')

    # add title and label
    plt.title(title)
    plt.xlabel('Token Number (k)', fontdict={'size': 14})
    plt.ylabel('Latency (s)', fontdict={'size': 14})

    # show axies
    plt.xlim(0, 32)
    plt.xticks(np.arange(0, 33, 4))
    plt.yticks(yticks)

    # legend
    plt.legend(loc='upper left')

    # show figure
    # plt.show()
    plt.tight_layout()
    plt.savefig(name)

plot_scartter(x, y1_1, y1_2, 'latency_compare_OPT-6.7B.jpg', np.arange(0, 5, 1), 'OPT-6.7B')
plot_scartter(x, y2_1, y2_2, 'latency_compare_OPT-13B.jpg', np.arange(0, 8, 2), 'OPT-13B')
plot_scartter(x, y3_1, y3_2, 'latency_compare_OPT-30B.jpg', np.arange(0, 5, 1), 'OPT-30B')
plot_scartter(x, y4_1, y4_2, 'latency_compare_Llama-2-7b-longlora.jpg', np.arange(0, 5, 1), 'Llama-2-7b-longlora')