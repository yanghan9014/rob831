import numpy as np
import matplotlib.pyplot as plt

data_usage = np.array([1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01])
mean_return = np.array([2320.2, 2094.7, 2874.9, 3106.9, 3326.1, 2935.1, 2374.1, 2746.2, 1778.2, 713.5, 351.7, 728.6])
std_return = np.array([1367.4, 1308.9, 1182.7, 767.2, 231.4, 1114.8, 1037.6, 595.5, 864.5, 213.1, 10.9, 97.6])

plt.figure(figsize=(10, 6))

plt.errorbar(data_usage, mean_return, yerr=std_return, fmt='-o', capsize=5, label='Mean Return with Std Dev')

plt.xlabel('Expert Data Usage')
plt.ylabel('Mean Return')
plt.title('Ablation Study: Expert Data Usage vs Return')

plt.grid(True)
plt.legend()
plt.savefig("ablation_data_usage.png")

