import numpy as np
import matplotlib.pyplot as plt

dagger_iterations = np.arange(1, 11)
mean_return_ant = np.array([2835.25830078125, 4333.7294921875, 4125.630859375, 4583.13525390625, 4718.44482421875, 4663.70458984375, 4736.9521484375, 4709.5693359375, 4750.33544921875, 4656.1064453125])
std_return_ant = np.array([1349.746337890625, 87.5212631225586, 991.4265747070312, 107.3659896850586, 70.81339263916016, 168.47076416015625, 77.7952880859375, 129.82826232910156, 26.30199432373047, 90.38716888427734])
mean_return_expert_ant = 4714
mean_return_bc_ant = 2941

mean_return_humanoid = np.array([314.79010009765625, 276.9959411621094, 326.8983459472656, 361.24212646484375, 443.5101013183594, 522.4112548828125, 994.6205444335938, 895.8815307617188, 1549.7454833984375, 1992.51025390625])
std_return_humanoid = np.array([77.18596649169922, 42.6093635559082, 47.54902267456055, 61.00497055053711, 132.1398468017578, 171.6398162841797, 470.5874938964844, 454.78826904296875, 691.607666015625, 1044.481201171875])
mean_return_expert_humanoid = 10365.5
mean_return_bc_humanoid = 316.6

fig, axs = plt.subplots(1, 2, figsize=(16, 7)) 

axs[0].errorbar(dagger_iterations, mean_return_ant, yerr=std_return_ant, fmt='-o', capsize=5, label='DAgger Policy (Ant)')
axs[0].axhline(y=mean_return_expert_ant, color='r', linestyle='--', label='Expert Policy (Ant)')
axs[0].axhline(y=mean_return_bc_ant, color='g', linestyle='--', label='Behavioral Cloning Policy (Ant)')
axs[0].set_xlabel('DAgger Iterations')
axs[0].set_ylabel('Mean Return')
axs[0].set_title('DAgger Learning Curve: Ant')
axs[0].legend()
axs[0].grid(True)

axs[1].errorbar(dagger_iterations, mean_return_humanoid, yerr=std_return_humanoid, fmt='-o', capsize=5, label='DAgger Policy (Humanoid)')
axs[1].axhline(y=mean_return_expert_humanoid, color='r', linestyle='--', label='Expert Policy (Humanoid)')
axs[1].axhline(y=mean_return_bc_humanoid, color='g', linestyle='--', label='Behavioral Cloning Policy (Humanoid)')
axs[1].set_xlabel('DAgger Iterations')
axs[1].set_ylabel('Mean Return')
axs[1].set_title('DAgger Learning Curve: Humanoid')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("dagger_learning_curve.png")