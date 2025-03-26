
import os
import numpy as np
import imageio
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from gymnasium_robotics.utils import rotations
from learn_utils.arguments import get_args

gym.register_envs(gymnasium_robotics)


def goal_distance(goal_a, goal_b, env_name):
    assert goal_a.shape == goal_b.shape
    if env_name == 'FetchPush-v1' or env_name == 'FetchPickAndPlace-v1' or env_name == 'FetchSlide-v1':
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    elif env_name == 'HandManipulateEggRotate-v1' or env_name == 'HandManipulateBlockRotateXYZ-v1' or env_name == 'HandManipulatePenRotate-v1':
        assert goal_a.shape[-1] == 7
        d_rot = np.zeros_like(goal_b[..., 0])
        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
        return d_rot
    else:
        return np.inf


if __name__ == "__main__":
    # get the params
    args = get_args()

    # Load expert datasets
    expert_path = 'RL_trained_agents/' + args.env_name + '/RL_expert_record_best_' + str(args.demo_length) + 'demos.npz'
    expert_data = np.load(expert_path)
    actions = expert_data['action']
    obs = expert_data['obs']
    desired_goals = expert_data['g'][:,0,:]
    initial_positions = expert_data['ag'][:,0,:]

    print(initial_positions.shape)
    print(desired_goals.shape)

    # Calculate distance
    distance = goal_distance(initial_positions, desired_goals, args.env_name)
    print(distance)

    max_dis_diff = 0
    bin_step = 0
    if args.env_name == 'FetchPush-v1':
        max_dis_diff = 0.52
        bin_step=0.01
    elif args.env_name == 'FetchPickAndPlace-v1':
        max_dis_diff = 0.62
        bin_step=0.01
    elif args.env_name == 'HandManipulateEggRotate-v1' or args.env_name == 'HandManipulateBlockRotateXYZ-v1':
        max_dis_diff = np.pi*2
        bin_step=0.1

    # Plot distribution
    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    morandi_blue_hist = "#85969C"  # Darker pastel blue for the histogram
    morandi_blue_pdf  = "#6F8B92"  # Slightly darker, complementary blue for the PDF line

	# Define your bins: from 0 to π in increments of 0.1
    bins = np.arange(0, max_dis_diff, bin_step)
    counts, bin_edges, patches = plt.hist(distance, bins=bins, color='steelblue', edgecolor='steelblue', alpha=0.6, \
            density=True)

    # Prepare data for KDE
    kde = gaussian_kde(distance)           # Fit the KDE model
    x_vals = np.linspace(0, max_dis_diff, 200)      # Points at which we evaluate the PDF
    pdf_vals = kde(x_vals)

    # Overlay the KDE PDF line
    plt.plot(x_vals, pdf_vals, color='steelblue', linewidth=2)

	# Labeling
    plt.xlabel('Goal distances ' + r'$d(g_{init}, g_d)$', fontsize=22)
    plt.ylabel('Histogram + PDF (KDE)', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()
