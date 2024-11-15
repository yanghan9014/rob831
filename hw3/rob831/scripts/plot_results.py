import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from matplotlib import pyplot as plt
import pdb

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

def results_stats(files):
    all_Y = []
    for file in files:
        X, Y = get_section_results(file)
        all_Y.append(Y)
    
    # Calculate the average for each metric at each step
    X = X[1:]
    avg_Y = [sum(y) / len(y) for y in zip(*all_Y)]
    std_Y = [np.std(y) for y in zip(*all_Y)]
    
    return X, avg_Y, std_Y

def write_averaged_results(logdir, X, avg_Y):
    # Create a new event file
    with tf.summary.FileWriter(logdir) as writer:
        for i, (x, y) in enumerate(zip(X, avg_Y)):
            summary = tf.Summary()
            summary.value.add(tag='Train_EnvstepsSoFar', simple_value=x)
            summary.value.add(tag='Train_AverageReturn', simple_value=y)
            writer.add_summary(summary, global_step=x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    logdir_dqn = os.path.join(args.logdir, 'q1_dqn*', 'events*')
    eventfiles_dqn = glob.glob(logdir_dqn)
    logdir_dou = os.path.join(args.logdir, 'q1_doubledqn*', 'events*')
    eventfiles_dou = glob.glob(logdir_dou)

    X, avg_Y_dqn, std_Y_dqn = results_stats(eventfiles_dqn)
    X, avg_Y_dou, std_Y_dou = results_stats(eventfiles_dou)

    plt.figure(figsize=(10, 6))
    plt.errorbar(X, avg_Y_dqn, yerr=std_Y_dqn, fmt='-o', capsize=5, label='DQN')
    plt.errorbar(X, avg_Y_dou, yerr=std_Y_dou, fmt='-o', capsize=5, label='Double DQN')
    plt.xlabel('Steps')
    plt.ylabel('Train_AverageReturn')
    plt.title('Average performance of DQN and DDQN')
    plt.legend()
    plt.show()

    # output_dir = 'q1_dqn_avg_LunarLander-v3'
    # os.makedirs(output_dir, exist_ok=True)
    # write_averaged_results(output_dir, X[1:], avg_Y)

