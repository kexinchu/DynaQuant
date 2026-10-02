import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

############### sharegpt
# read excel file
df = pd.read_excel('preprocessing latency VS overall latency.xlsx', sheet_name='ShareGPT')

# import data
x = [a[:-1] for a in df.iloc[:, 0]]
y1 = df['OPT 6.7B']
y2 = df['OPT 13B']
y3 = df['LLaMA-2 7B']
y4 = df['LLaMA-2 13B']

y1 = y1 / (1-y1)
y2 = y2 / (1-y2)
y3 = y3 / (1-y3)
y4 = y4 / (1-y4)


plt.figure(figsize=(8, 4))

# plot
plt.plot(x, y1, 'o-', label='OPT 6.7B')
plt.plot(x, y2, 'o-',label='OPT 13B')
plt.plot(x, y3, 'o-',label='LLaMA-2 7B')
plt.plot(x, y4, 'o-',label='LLaMA-2 13B')
# # dot
# plt.scatter(x, y1)
# plt.scatter(x, y2)
# plt.scatter(x, y3)
# plt.scatter(x, y4)

# add title and label
# plt.title('Line Graph of Data')
plt.xlabel('Request Length (k)', fontdict={'size': 14})
plt.ylabel('TTFT / TTET', fontdict={'size': 14})

# show axies
plt.xticks(np.arange(0, 31, 3))
plt.yticks(np.arange(0, 1.3, 0.2))
# plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 


# legend
plt.legend(loc='lower right')

# show figure
# plt.show()
plt.tight_layout()
plt.savefig('sharegpt.jpg')

################ LongAlign
# read excel file
df = pd.read_excel('preprocessing latency VS overall latency.xlsx', sheet_name='LongAlign')

# import data
x = [a[:-1] for a in df.iloc[:, 0]]
y1 = df['OPT 6.7B']
y2 = df['OPT 13B']
y3 = df['LLaMA-2 7B']
y4 = df['LLaMA-2 13B']

y1 = y1 / (1-y1)
y2 = y2 / (1-y2)
y3 = y3 / (1-y3)
y4 = y4 / (1-y4)

plt.figure(figsize=(8, 4))

# plot
plt.plot(x, y1, 'o-', label='OPT 6.7B')
plt.plot(x, y2, 'o-', label='OPT 13B')
plt.plot(x, y3, 'o-', label='LLaMA-2 7B')
plt.plot(x, y4, 'o-', label='LLaMA-2 13B')
# # dot
# plt.scatter(x, y1)
# plt.scatter(x, y2)
# plt.scatter(x, y3)
# plt.scatter(x, y4)

# add title and label
# plt.title('Line Graph of Data')
plt.xlabel('Request Length (k)', fontdict={'size': 14})
plt.ylabel('TTFT / TTET', fontdict={'size': 14})

# show axies
plt.xticks([-1] + [a for a in range(0, 31, 3)])
plt.yticks(np.arange(0, 1.2, 0.2))
# plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 

# legend
plt.legend(loc='lower right')

# show figure
# plt.show()
plt.tight_layout()
plt.savefig('LongAlign.jpg')