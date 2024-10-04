# import numpy as np
# import matplotlib.pyplot as plt

# # Example data
# dagger_iterations = np.arange(1, 11)  # Replace with your DAgger iterations
# mean_return_ant = np.array([2835.25830078125, 4333.7294921875, 4125.630859375, 4583.13525390625, 4718.44482421875, 4663.70458984375, 4736.9521484375, 4709.5693359375, 4750.33544921875, 4656.1064453125])
# std_return_ant = np.array([1349.746337890625, 87.5212631225586, 991.4265747070312, 107.3659896850586, 70.81339263916016, 168.47076416015625, 77.7952880859375, 129.82826232910156, 26.30199432373047, 90.38716888427734])
# mean_return_expert_ant = 4714  # Replace with mean return of expert policy
# mean_return_bc_ant = 2941  # Replace with mean return of behavioral cloning agent

# mean_return_humanoid = np.array([314.79010009765625, 276.9959411621094, 326.8983459472656, 361.24212646484375, 443.5101013183594, 522.4112548828125, 994.6205444335938, 895.8815307617188, 1549.7454833984375, 1992.51025390625])
# std_return_humanoid = np.array([77.18596649169922, 42.6093635559082, 47.54902267456055, 61.00497055053711, 132.1398468017578, 171.6398162841797, 470.5874938964844, 454.78826904296875, 691.607666015625, 1044.481201171875])
# mean_return_expert_humanoid = 10365.5  # Replace with mean return of expert policy
# mean_return_bc_humanoid = 316.6  # Replace with mean return of behavioral cloning agent

# # Create subplots
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns for side-by-side subplots

# # Left subplot for Ant environment
# axs[0].errorbar(dagger_iterations, mean_return_ant, yerr=std_return_ant, fmt='-o', capsize=5, label='DAgger Policy (Ant)')
# axs[0].axhline(y=mean_return_expert_ant, color='r', linestyle='--', label='Expert Policy (Ant)')
# axs[0].axhline(y=mean_return_bc_ant, color='g', linestyle='--', label='Behavioral Cloning Policy (Ant)')
# axs[0].set_xlabel('DAgger Iterations')
# axs[0].set_ylabel('Mean Return')
# axs[0].set_title('DAgger Learning Curve: Ant')
# axs[0].legend()
# axs[0].grid(True)

# # Right subplot for Humanoid environment
# axs[1].errorbar(dagger_iterations, mean_return_humanoid, yerr=std_return_humanoid, fmt='-o', capsize=5, label='DAgger Policy (Humanoid)')
# axs[1].axhline(y=mean_return_expert_humanoid, color='r', linestyle='--', label='Expert Policy (Humanoid)')
# axs[1].axhline(y=mean_return_bc_humanoid, color='g', linestyle='--', label='Behavioral Cloning Policy (Humanoid)')
# axs[1].set_xlabel('DAgger Iterations')
# axs[1].set_ylabel('Mean Return')
# axs[1].set_title('DAgger Learning Curve: Humanoid')
# axs[1].legend()
# axs[1].grid(True)

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.savefig("dagger_learning_curve.png")
import numpy as np
import matplotlib.pyplot as plt

# Expert data usage (x-axis) values
data_usage = np.array([1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01])

# Mean return values (y-axis) corresponding to expert data usage
mean_return = np.array([2320.2, 2094.7, 2874.9, 3106.9, 3326.1, 2935.1, 2374.1, 2746.2, 1778.2, 713.5, 351.7, 728.6])

# Standard deviation values for each mean return
std_return = np.array([1367.4, 1308.9, 1182.7, 767.2, 231.4, 1114.8, 1037.6, 595.5, 864.5, 213.1, 10.9, 97.6])

# Create the plot
plt.figure(figsize=(10, 6))

# Plot with error bars
plt.errorbar(data_usage, mean_return, yerr=std_return, fmt='-o', capsize=5, label='Mean Return with Std Dev')
# axs[1].axhline(y=mean_return_expert_humanoid, color='r', linestyle='--', label='Expert Policy (Humanoid)')
# axs[1].axhline(y=mean_return_bc_humanoid, color='g', linestyle='--', label='Behavioral Cloning Policy (Humanoid)')

# Labels and title
plt.xlabel('Expert Data Usage')
plt.ylabel('Mean Return')
plt.title('Ablation Study: Expert Data Usage vs Return')

# Show grid for better readability
plt.grid(True)

# Show the plot
plt.legend()
plt.savefig("ablation_data_usage.png")

