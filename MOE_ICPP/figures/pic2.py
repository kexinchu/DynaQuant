import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

############### swap_in
# read excel file
df = pd.read_excel('swap_latency.xlsx', sheet_name='swap_in')

# import data
x = df['token number'].map(lambda a: pd.to_numeric(a[:-1]))
y1 = df['recompute latency']
y2 = df['swap in latency']

plt.figure(figsize=(8, 4))

# plot
width = 0.15
plt.bar(x-width, y1, width*2, label='recompute latency')
plt.bar(x+width, y2, width*2, label='swap-in latency')

# add title and label
# plt.title('Line Graph of Data')
plt.xlabel('Token Number (k)', fontdict={'size': 14})
plt.ylabel('Latency (s)', fontdict={'size': 14})

# show axies
plt.xlim(0, 32)
plt.xticks(x)
plt.yticks(np.arange(0, 7, 1))

# legend
plt.legend(loc='upper left')

# show figure
# plt.show()
plt.tight_layout()
plt.savefig('swap_in.jpg')

################ sparsification
# read excel file
df = pd.read_excel('swap_latency.xlsx', sheet_name='sparsification')

# import data
x = df['LLM-Layer'] 
y1 = df['OPT-6.7B'] 
y2 = df['OPT-13B'] 
y3 = df['OPT-30B'] 
y4 = df['Llama-7B'] 
y5 = df['Llama-13B'] 

plt.figure(figsize=(8, 4))

# plot
plt.plot(x, y1, '.-', label='OPT-6.7B')
plt.plot(x, y2, '.-', label='OPT-13B')
plt.plot(x, y3, '.-', label='OPT-30B')
plt.plot(x, y4, '.-', label='Llama-7B')
plt.plot(x, y4, '.-', label='Llama-13B')
# # dot
# plt.scatter(x, y1, 10)
# plt.scatter(x, y2, 10)
# plt.scatter(x, y3, 10)
# plt.scatter(x, y4, 10)

# add title and label
# plt.title('Line Graph of Data')
plt.xlabel('LLM Layer', fontdict={'size': 14})
plt.ylabel('Percentage', fontdict={'size': 14})

# show axies
plt.xticks(np.arange(0, 50, 5))
plt.yticks(np.arange(0, 1.1, 0.2))
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 

# legend
plt.legend(loc='lower right')

# show figure
# plt.show()
plt.tight_layout()
plt.savefig('sparsification.jpg')