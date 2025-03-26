import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from learn_utils.arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()

    eval_file_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_her.npy'
    eval_file_1_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_lfd_ddpg+her_suboptimal_demo' + str(args.demo_length) + '.npy'
    eval_file_2_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_goalgail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_rewardD_demo' + str(args.demo_length) + '.npy'
    eval_file_3_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_goalsgail_new_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_matchdislimit' + str(args.match_dis_limit) + '_rewardD_demo' + str(args.demo_length) + '.npy'
    # eval_file_4_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_goalsail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo' + str(args.demo_length) + '.npy'
    # eval_file_5_path = args.save_dir + args.env_name + '/seed_' + str(args.seed) + '/eval_success_rates_goalsail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo' + str(args.demo_length) + '.npy'

    show_length = 100
    if not os.path.isfile(eval_file_path):
        print(eval_file_path)
        print("Result file do not exist!")
    else:
        data = np.load(eval_file_path)[:show_length]
        data1 = np.load(eval_file_1_path)[:show_length]
        data2 = np.load(eval_file_2_path)[:show_length]
        data3 = np.load(eval_file_3_path)[:show_length]
        # data4 = np.load(eval_file_4_path)[:show_length]
        # data5 = np.load(eval_file_5_path)[:show_length]
        print(data)

        x = np.linspace(0, len(data), len(data))
        x1 = np.linspace(0, len(data1), len(data1))
        x2 = np.linspace(0, len(data2), len(data2))
        x3 = np.linspace(0, len(data3), len(data3))
        # x4 = np.linspace(0, len(data4), len(data4))
        # x5 = np.linspace(0, len(data5), len(data5))

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test success rate', fontsize=16)
        plt.title(args.env_name, fontsize=20)

        plt.plot(x, data, color='purple', linewidth=2, label='HER, RL baseline')
        plt.plot(x1, data1, color='green', linewidth=2, label='HER with demo')
        plt.plot(x2, data2, color='blue', linewidth=2, label='goal-GAIL')
        plt.plot(x3, data3, color='red', linewidth=2, label='goal-SAGAIL')
        # plt.plot(x4, data4, color='orange', linewidth=2, label='goal-SAIL, demo200_30%, no match')
        # plt.plot(x5, data5, color='brown', linewidth=2, label='goal-SAIL, demo200_30%, no match')

        plt.legend(loc='lower right')
        plt.show()
