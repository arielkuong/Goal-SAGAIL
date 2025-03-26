import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy import stats
from learn_utils.arguments import get_args

seed_list = [123, 439, 516, 758, 835]

if __name__ == "__main__":

    args = get_args()

    eval_file_path_1b = args.save_dir + args.env_name + '/seed_' + str(seed_list[0]) + '/eval_success_rates_her.npy'
    eval_file_path_2b = args.save_dir + args.env_name + '/seed_' + str(seed_list[1]) + '/eval_success_rates_her.npy'
    eval_file_path_3b = args.save_dir + args.env_name + '/seed_' + str(seed_list[2]) + '/eval_success_rates_her.npy'
    eval_file_path_4b = args.save_dir + args.env_name + '/seed_' + str(seed_list[3]) + '/eval_success_rates_her.npy'
    eval_file_path_5b = args.save_dir + args.env_name + '/seed_' + str(seed_list[4]) + '/eval_success_rates_her.npy'

    #eval_file_path_1 = args.save_dir + args.env_name + '/seed_' + str(seed[0]) + '/eval_success_rates_lfd_ddpg+her_suboptimal_allsuccessdemo_demoweight0.3_demo200.npy'
    #eval_file_path_2 = args.save_dir + args.env_name + '/seed_' + str(seed[1]) + '/eval_success_rates_lfd_ddpg+her_suboptimal_allsuccessdemo_demoweight0.3_demo200.npy'
    #eval_file_path_3 = args.save_dir + args.env_name + '/seed_' + str(seed[2]) + '/eval_success_rates_lfd_ddpg+her_suboptimal_allsuccessdemo_demoweight0.3_demo200.npy'
    # eval_file_path_4 = args.save_dir + args.env_name + '/seed_' + str(seed[3]) + '/eval_success_rates.npy'
    # eval_file_path_5 = args.save_dir + args.env_name + '/seed_' + str(seed[4]) + '/eval_success_rates.npy'

    # eval_file_path_1 = args.save_dir + args.env_name + '/seed_' + str(seed[0]) + '/eval_success_rates_goalgail_suboptimal_allsuccessdemo_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    # eval_file_path_2 = args.save_dir + args.env_name + '/seed_' + str(seed[1]) + '/eval_success_rates_goalgail_suboptimal_allsuccessdemo_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    # eval_file_path_3 = args.save_dir + args.env_name + '/seed_' + str(seed[2]) + '/eval_success_rates_goalgail_suboptimal_allsuccessdemo_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    # eval_file_path_4 = args.save_dir + args.env_name + '/seed_' + str(seed[3]) + '/eval_success_rates_goalgail_suboptimal_allsuccessdemo_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    # eval_file_path_5 = args.save_dir + args.env_name + '/seed_' + str(seed[4]) + '/eval_success_rates_goalgail_suboptimal_allsuccessdemo_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'

    eval_file_path_1 = args.save_dir + args.env_name + '/seed_' + str(seed_list[0]) + '/eval_success_rates_goalsgail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    eval_file_path_2 = args.save_dir + args.env_name + '/seed_' + str(seed_list[1]) + '/eval_success_rates_goalsgail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    eval_file_path_3 = args.save_dir + args.env_name + '/seed_' + str(seed_list[2]) + '/eval_success_rates_goalsgail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    eval_file_path_4 = args.save_dir + args.env_name + '/seed_' + str(seed_list[3]) + '/eval_success_rates_goalsgail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'
    eval_file_path_5 = args.save_dir + args.env_name + '/seed_' + str(seed_list[4]) + '/eval_success_rates_goalsgail_suboptimal_allsuccessdemo_eprlimit-2_rewardD_demoweight0.3_demo'+ str(args.demo_length) + '.npy'


    data_len = 20
    data1 = np.load(eval_file_path_1)[:data_len]
    data2 = np.load(eval_file_path_2)[:data_len]
    data3 = np.load(eval_file_path_3)[:data_len]
    data4 = np.load(eval_file_path_4)[:data_len]
    data5 = np.load(eval_file_path_5)[:data_len]

    data1b = np.load(eval_file_path_1b)[:data_len]
    data2b = np.load(eval_file_path_2b)[:data_len]
    data3b = np.load(eval_file_path_3b)[:data_len]
    data4b = np.load(eval_file_path_4b)[:data_len]
    data5b = np.load(eval_file_path_5b)[:data_len]

    # print(data1b)
    # print(data2b)
    # print(data3b)
    # print(data4b)
    # print(data5b)

    diff_data1 = data1 - data1b
    diff_data2 = data2 - data2b
    diff_data3 = data3 - data3b
    diff_data4 = data4 - data4b
    diff_data5 = data5 - data5b

    # diff_data1 = np.divide(data1 - data1b, data1b)
    # diff_data2 = np.divide(data2 - data2b, data2b)
    # diff_data3 = np.divide(data3 - data3b, data3b)
    # diff_data4 = np.divide(data4 - data4b, data4b)
    # diff_data5 = np.divide(data5 - data5b, data5b)

    indice_size = 5
    indices = np.arange(0, len(diff_data1), indice_size)
    print(indices)
    diff_data1_sum = np.add.reduceat(diff_data1, indices)/indice_size
    diff_data2_sum = np.add.reduceat(diff_data2, indices)/indice_size
    diff_data3_sum = np.add.reduceat(diff_data3, indices)/indice_size
    diff_data4_sum = np.add.reduceat(diff_data4, indices)/indice_size
    diff_data5_sum = np.add.reduceat(diff_data5, indices)/indice_size

    # diff_data1_sum = diff_data1_sum*100
    # diff_data2_sum = diff_data2_sum*100
    # diff_data3_sum = diff_data3_sum*100
    # diff_data4_sum = diff_data4_sum*100
    # diff_data5_sum = diff_data5_sum*100

    print(diff_data1_sum.round(2))
    print(diff_data2_sum.round(2))
    print(diff_data3_sum.round(2))
    print(diff_data4_sum.round(2))
    print(diff_data5_sum.round(2))

    # x_ticks = np.linspace(0, len(diff_data1_sum), len(diff_data1_sum)+1).astype(int)
    diff_data_comb = np.array([diff_data1_sum, diff_data2_sum, diff_data3_sum, diff_data4_sum, diff_data5_sum])
    # print(diff_data_comb)
    print(diff_data_comb.mean(0).round(2))
    print(diff_data_comb.std(0).round(2))
    # print(x_ticks)
    #
    # mpl.style.use('ggplot')
    # fig = plt.figure(1)
    # fig.patch.set_facecolor('white')
    # plt.xlabel('Learning Epochs', fontsize=16)
    # # plt.ylabel('Success Rate Difference', fontsize=16)
    # plt.title('Success Rate Differnce (avgeraged across every 5 epochs)', fontsize=16)
    #
    # plt.imshow(diff_data_comb, cmap='bwr', vmin=-0.2, vmax=0.2)
    # plt.xticks(x_ticks, x_ticks*indice_size)
    # plt.yticks(np.arange(5),['seed1', 'seed2', 'seed3', 'seed4', 'seed5'], fontsize=16)
    # plt.colorbar(location='top')
    # for i in range(len(indices)):
    #     for j in range(len(seed)):
    #         text = plt.text(i, j, np.round(diff_data_comb[j, i],2),
    #                        ha="center", va="center", color="black")
    # plt.show()

    # one-sample t-Test
    p_values = []
    for stage in range(int(data_len/indice_size)):
        collection = []
        # print(stage*indice_size)
        # print((stage + 1)*indice_size)
        collection.append(diff_data1[stage*indice_size:(stage + 1)*indice_size])
        collection.append(diff_data2[stage*indice_size:(stage + 1)*indice_size])
        collection.append(diff_data3[stage*indice_size:(stage + 1)*indice_size])
        collection.append(diff_data4[stage*indice_size:(stage + 1)*indice_size])
        collection.append(diff_data5[stage*indice_size:(stage + 1)*indice_size])
        collection = np.array(collection)
        collection = collection.flatten()
        # print(collection.shape)
        res = stats.ttest_1samp(collection, popmean=0)
        # print(res.pvalue.round(3))
        p_values.append(res.pvalue.round(3))
    p_values = np.array(p_values)
    print(p_values)
