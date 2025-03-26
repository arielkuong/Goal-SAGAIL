import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from learn_utils.arguments import get_args
import os

if __name__ == "__main__":

    args = get_args()
    eval_file_path_1_mean = args.save_dir + args.env_name + '/Average_result_her_5seeds/data_mean.npy'
    eval_file_path_1_high = args.save_dir + args.env_name + '/Average_result_her_5seeds/data_high.npy'
    eval_file_path_1_low = args.save_dir + args.env_name + '/Average_result_her_5seeds/data_low.npy'

    eval_file_path_2_mean = args.save_dir + args.env_name + '/Average_result_lfd_ddpg+her_suboptimal_demo' + str(args.demo_length) +'_5seeds/data_mean.npy'
    eval_file_path_2_high = args.save_dir + args.env_name + '/Average_result_lfd_ddpg+her_suboptimal_demo' + str(args.demo_length) +'_5seeds/data_high.npy'
    eval_file_path_2_low = args.save_dir + args.env_name + '/Average_result_lfd_ddpg+her_suboptimal_demo' + str(args.demo_length) +'_5seeds/data_low.npy'

    eval_file_path_3_mean = args.save_dir + args.env_name + '/Average_result_goalgail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_demo' + str(args.demo_length) +'_5seeds/data_mean.npy'
    eval_file_path_3_high = args.save_dir + args.env_name + '/Average_result_goalgail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_demo' + str(args.demo_length) +'_5seeds/data_high.npy'
    eval_file_path_3_low = args.save_dir + args.env_name + '/Average_result_goalgail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_demo' + str(args.demo_length) +'_5seeds/data_low.npy'

    eval_file_path_4_mean = args.save_dir + args.env_name + '/Average_result_goalsagail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_matchdislimit' + str(args.match_dis_limit) + '_demo' + str(args.demo_length) +'_5seeds/data_mean.npy'
    eval_file_path_4_high = args.save_dir + args.env_name + '/Average_result_goalsagail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_matchdislimit' + str(args.match_dis_limit) + '_demo' + str(args.demo_length) +'_5seeds/data_high.npy'
    eval_file_path_4_low = args.save_dir + args.env_name + '/Average_result_goalsagail_suboptimal_gailweight' + str(args.gail_weight) + 'anneal_matchdislimit' + str(args.match_dis_limit) + '_demo' + str(args.demo_length) +'_5seeds/data_low.npy'


    data_len = 100
    data1_mean = np.load(eval_file_path_1_mean)[:data_len]
    data1_high = np.load(eval_file_path_1_high)[:data_len]
    data1_low = np.load(eval_file_path_1_low)[:data_len]
    data2_mean = np.load(eval_file_path_2_mean)[:data_len]
    data2_high = np.load(eval_file_path_2_high)[:data_len]
    data2_low = np.load(eval_file_path_2_low)[:data_len]
    data3_mean = np.load(eval_file_path_3_mean)[:data_len]
    data3_high = np.load(eval_file_path_3_high)[:data_len]
    data3_low = np.load(eval_file_path_3_low)[:data_len]
    data4_mean = np.load(eval_file_path_4_mean)[:data_len]
    data4_high = np.load(eval_file_path_4_high)[:data_len]
    data4_low = np.load(eval_file_path_4_low)[:data_len]

    data_demo = np.repeat(0.6, data1_mean.shape[0])
    print(data_demo)

    x = np.linspace(0, data1_mean.shape[0], data1_mean.shape[0])

    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Test Success Rate', fontsize=20)
    plt.title(args.env_name, fontsize=20)

    plt.plot(x, data1_mean, color='purple', linewidth=2, label='HER(RL baseline)')
    plt.fill_between(x, data1_low, data1_high, color='purple', alpha=0.1)
    plt.plot(x, data_demo, color='brown', linewidth=1, label='expert')
    plt.plot(x, data2_mean, color='blue', linewidth=2, label='DDPGfD+HER')
    plt.fill_between(x, data2_low, data2_high, color='blue', alpha=0.1)
    plt.plot(x, data3_mean, color='green', linewidth=2, label='Goal_GAIL')
    plt.fill_between(x, data3_low, data3_high, color='green', alpha=0.1)
    plt.plot(x, data4_mean, color='red', linewidth=2, label='Goal_SAGAIL(ours)')
    plt.fill_between(x, data4_low, data4_high, color='red', alpha=0.1)

    plt.legend(loc='lower right')
    plt.legend(fontsize=18)

    plt.show()
