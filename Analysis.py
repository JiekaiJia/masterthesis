import copy

import matplotlib.pyplot as plt
import numpy as np

# Actions frequency means the number of steps between two consecutive steps.
actions_frequency = (1, 2, 3, 4, 5, 6)
# Mean episode reward that based on different action frequency
rewards = {
    'name': 'rewards',
    'ShortQPolicy': (-33.916666666666664, -91.45833333333333, -118.575, -137.90833333333333, -152.33333333333334, -164.29166666666666),
    # 'RandomPolicy1': (-89.91666666666667, -112.63333333333334, -141.10833333333332, -169.06666666666666, -195.54166666666666, -222.725),
    # 'RandomPolicy2': (-67.35833333333333, -74.65833333333333, -80.825, -94.19166666666666, -112.65, -128.95),
    'ppo_256_tanh': (-47.475, -55.975, -70.89166666666667, -87.25, -109.11666666666666, -135.45),
    'belief_ppo_256_tanh': (-62.11666666666667, -66.93333333333334, -77.08333333333333, -87.29166666666667, -101.81666666666666, -123.84166666666667),
}

std_rewards = {
    'ShortQPolicy': ((-35.89809494603521, -31.93523838729812), (-94.13303365232683, -88.78363301433983), (-121.48412736575077, -115.66587263424924), (-141.09081198095004, -134.72585468571663), (-155.63025349591064, -149.03641317075605), (-167.43930694515734, -161.14402638817597)),
    # 'RandomPolicy1': ((-92.62388350136025, -87.20944983197309), (-115.34605673634981, -109.92060993031687), (-144.18034467551735, -138.0363219911493), (-172.72145335748942, -165.4118799758439), (-198.92721753615317, -192.15611579718015), (-225.8039896765527, -219.64601032344729)),
    # 'RandomPolicy2': ((-69.82607540986714, -64.89059125679952), (-77.57216020869626, -71.7445064579704), (-83.35111684473584, -78.29888315526416), (-97.11450669167843, -91.2688266416549), (-115.61289815983744, -109.68710184016257), (-131.55564265146262, -126.34435734853734)),
    'ppo_256_tanh': ((-49.80070849565554, -45.14929150434446), (-58.5627321529814, -53.387267847018606), (-73.6385735279123, -68.14475980542103), (-89.9919500633857, -84.5080499366143), (-111.52977380607702, -106.7035595272563), (-138.18910474954544, -132.71089525045454)),
    'belief_ppo_256_tanh': ((-64.63325737410686, -59.600075959226466), (-69.54138616550848, -64.32528050115819), (-79.88024087234683, -74.28642579431983), (-89.85615697144927, -84.72717636188408), (-104.35507055467464, -99.27826277865869), (-126.45731594669876, -121.22601738663458)),
}

episode_pkg_drop = {
    'name': 'episode_pkg_drop',
    'JSQ': (0.04870370370370371, 0.07694444444444444, 0.1262037037037037, 0.18675925925925926, 0.24888888888888888, 0.31379629629629624),
    # 'baseline': (0.07712962962962962, 0.08185185185185186, 0.10097222222222223, 0.12324074074074073, 0.14564814814814814, 0.18157407407407408),
    'communication model': (0.05513888888888889, 0.07462962962962963, 0.09472222222222221, 0.12254629629629629, 0.15560185185185185, 0.19060185185185186),
}

std_episode_pkg_drop = {
    'JSQ': ((0.04160319432074347, 0.055804213086663945), (0.07030028306752989, 0.08358860582135899), (0.11950045448676945, 0.13290695292063798), (0.17925101181008563, 0.1942675067084329), (0.24132731361751497, 0.2564504641602628), (0.30596671136509146, 0.321625881227501)),
    # 'baseline': ((0.0696146132830983, 0.08464464597616093), (0.07483055330665456, 0.08887315039704916), (0.09309657954637544, 0.10884786489806901), (0.11509812419055837, 0.1313833572909231), (0.13888323676678846, 0.15241305952950782), (0.17356154689987047, 0.1895866012482777)),
    'communication model': ((0.0491218526228569, 0.06115592515492088), (0.06784776572536663, 0.08141149353389263), (0.08791277862636446, 0.10153166581807996), (0.11496044550914043, 0.13013214708345214), (0.14754238466668268, 0.16366131903702102), (0.18285188522627174, 0.19835181847743197)),
}

heat_map = {
    "JSQ": [[0.47729673, 1., 1., 1., 1., 1.],
            [0., 0.47680412, 1., 1., 1., 1.],
            [0., 0., 0.49046322, 1., 1., 1.],
            [0., 0., 0., 0.50326797, 1., 1.],
            [0., 0., 0., 0., 0.5245098, 1.],
            [0., 0., 0., 0., 0., 0.5]],
    "DVPENet": [[0.36257468, 0.46287768, 0.60350266, 0.72738794, 0.82590039, 0.85717665],
                 [0.28974093, 0.37040206, 0.51675002, 0.66616366, 0.78220484, 0.83530224],
                 [0.2175573,  0.27485968, 0.41623282, 0.58323447, 0.69040651, 0.77754949],
                 [0.16393976, 0.21016972, 0.32017577, 0.47679563, 0.61995354, 0.69553846],
                 [0.13644317, 0.16334905, 0.25329822, 0.38385971, 0.54261357, 0.6335073 ],
                 [0.12612335, 0.13371035, 0.20943439, 0.31732969, 0.43329694, 0.52308602]],
    "DVPENet_d":[[0.39539207, 0.46864956, 0.57756828, 0.66421457, 0.73812509, 0.7562945 ],
                 [0.32699586, 0.39542893, 0.54022564, 0.60716157, 0.69363557, 0.71029685],
                 [0.26625887, 0.31425315, 0.43867499, 0.54506276, 0.61782234, 0.68934805],
                 [0.20296479, 0.25256517, 0.34699399, 0.44913325, 0.54968743, 0.56778525],
                 [0.1721646,  0.19588336, 0.29360229, 0.39263768, 0.47700197, 0.52843391],
                 [0.1355677,  0.16636598, 0.22940624, 0.32558226, 0.41784843, 0.45673619]],
    "baseline_real":[[0.3340028,  0.46418427, 0.60028476, 0.71618703 ,0.80404297 ,0.86714235],
 [0.2890256,  0.40530242, 0.5368192 , 0.65928569, 0.7609962 , 0.83975052],
 [0.25049393, 0.34946084 ,0.47697815, 0.6034612  ,0.71930262 ,0.8049064 ],
 [0.21903613, 0.30414068 ,0.42400942, 0.55317677, 0.67092955 ,0.76819644],
 [0.18873143 ,0.26170609, 0.37094127, 0.49803284 ,0.62097846 ,0.7272121 ],
 [0.15939827 ,0.22152315 ,0.31713931, 0.44020106 ,0.56509319 ,0.6816723 ]],
    "baseline_delay":[[0.74830422, 0.76662392, 0.78665948, 0.80893643, 0.82187078, 0.84082777],
 [0.68427139 ,0.70131427 ,0.72296738 ,0.75316697, 0.78401095, 0.80081321],
 [0.59585517, 0.62073846, 0.65822022 ,0.67257768, 0.71914856, 0.72195229],
 [0.51170525 ,0.53461289, 0.57377004 ,0.60393031 ,0.64216299, 0.64763669],
 [0.44494683 ,0.46005253, 0.48925345 ,0.52759681 ,0.57111159, 0.58855111],
 [0.36648319 ,0.372148  , 0.41500361 ,0.43678528 ,0.45490884, 0.50673442]],
    "DVPENet_dis":[[0.12175325, 0.33333333, 0.75496689, 0.99145299, 1.,         1.        ],
 [0.04   ,    0.13480392 ,0.52380952, 0.94863014 ,1. ,        1.        ],
 [0.00597015, 0.03514377 ,0.22028986 ,0.75193798 ,0.97379913 ,1.        ],
 [0.         ,0.00416667 ,0.04201681 ,0.56133829 ,0.84647303, 0.99300699],
 [0.         ,0. ,        0.01142857 ,0.15       ,0.67515924 ,0.944     ],
 [0.         ,0.,         0.         ,0.01369863, 0.20588235 ,0.68316832]],
    "baseline_real_dis":[[0.     ,    0.105 ,     1.     ,    1.  ,       1.  ,       1.        ],
 [0.   ,      0.     ,    0.9109589  ,1.    ,     1.         ,1.        ],
 [0.    ,     0.      ,   0.17320261, 1.     ,    1.        , 1.        ],
 [0.         ,0.       ,  0.       ,  1.      ,   1.       ,  1.        ],
 [0.         ,0.        , 0.      ,   0.42533937, 1.      ,  1.        ],
 [0.     ,    0.         ,0.     ,    0.         ,1.        , 1.        ]],
    "baseline_delay_dis":[[0.96695652, 0.93573265, 0.93370166, 0.97526502 ,0.94623656 ,0.95857988],
 [0.91666667 ,0.90909091, 0.92690058, 0.9382716,  0.93956044 ,0.94805195],
 [0.80169972 ,0.83024691 ,0.84664537 ,0.87037037, 0.90419162 ,0.96      ],
 [0.67315175, 0.68273092, 0.76818182, 0.82325581 ,0.91666667, 0.8671875 ],
 [0.47058824, 0.51851852 ,0.6039604  ,0.75238095 ,0.80536913 ,0.77669903],
 [0.375    ,  0.48591549, 0.63333333, 0.66906475, 0.76635514 ,0.87692308]]
}

comm = {
    'name': 'communicatoin rate',
    'belief_ppo_256_tanh': (0.39276007041242966, 0.39721575911522117, 0.39662377661001924, 0.3919444739565517, 0.3898756412835674, 0.38412056740481015)
}

std_comm = {
    'belief_ppo_256_tanh': ((0.38705239914274125, 0.39846774168211807), (0.3914883949438303, 0.402943123286612), (0.39107997944850087, 0.4021675737715376), (0.38677820398825996, 0.3971107439248434), (0.38482828394759033, 0.39492299861954444), (0.37878901846873664, 0.38945211634088367))
}


def plot(dic, std_dic):
    plt.figure(figsize=(10, 5))
    # Plot mean of rewards.
    labels = []
    for k, v in dic.items():
        if k == 'name':
            continue
        labels.append(k)
        plt.plot(actions_frequency, v)
        # Plot confidence interval
        r_low, r_high = [], []
        for x, y in std_dic[k]:
            r_low.append(x)
            r_high.append(y)
        plt.fill_between(actions_frequency, r_low, r_high, alpha=0.3)

    plt.xlabel(r'$\Delta T$')
    plt.grid()
    # plt.ylabel(f"Average episode {dic['name']}")
    plt.ylabel(f"Average packet drop rate")
    # plt.legend(labels)
    # plt.title(f"Mean episode {dic['name']} vs. No. of actions_frequency")


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == '__main__':
    # plot(rewards, std_rewards)
    # plot(comm, std_comm)
    # plt.savefig("../myfigure/deltatcomm", dpi=300)
    # plot(comm, std_comm)
    # plt.show()
    # loss = pd.read_csv('./runs/encoder_loss.csv')
    # loss.plot(x='Step', y='Value')
    # plt.ylabel("Loss")
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(-0.5, 5.5)
    a = np.asarray(heat_map["baseline_delay_dis"])
    im = ax.imshow(a, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    fig.colorbar(im)
    plt.xlabel(r'$b_1$')
    plt.ylabel(r'$b_2$')
    plt.savefig("../myfigure/baseline_delay_dis_heatmap", dpi=300)
    plt.show()
    # x = np.arange(-3, 3, 0.1)
    # fig = plt.figure(figsize=(7, 4))
    # ax = fig.add_subplot(111)
    # ax.set_ylim(0, 1)
    # ax.grid()
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # ax.set_xlabel(r'$x$', fontsize=12)
    # ax.set_ylabel(r'$y$', fontsize=12)
    # plt.plot(x, sigmoid(x))
    # plt.plot(x, relu(x))
    # plt.plot(x, tanh(x))
    # plt.savefig("../myfigure/sigmoid", dpi=300)
    # plt.show()



# """"""
# #  -*- coding: utf-8 -*-
# # date: 2021
# # author: AllChooseC
#
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import signal
#
#
# matplotlib.use('TkAgg')
#
#
# def plot_metric(metric_values):
#     """Plot metric values in a line graph."""
#     plt.plot(metric_values, '-x')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.title('Accuracy vs. No. of epochs')
#
#
# def plot_losses(train_losses, vld_losses):
#     plt.plot(train_losses, '-x')
#     plt.plot(vld_losses, '-x')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['Training', 'Validation'])
#     plt.title('Loss vs. No. of epochs')
#
#
# def display_signal(data_loader):
#     """Display signals."""
#     count = 0
#     classes = ['N', 'A', 'O', '~']
#     for xs, ys in data_loader:
#         batch_size = xs.shape[0]
#         xs = xs.numpy()
#         ys = ys.numpy()
#         plt.figure(figsize=(15, 10))
#         for i in range(batch_size):
#             if count < 4:
#                 count += 1
#                 ax = plt.subplot(2, 2, count)
#                 tmp = np.squeeze(xs[i])
#                 t = (len(tmp) - 1) / 300
#                 t = np.linspace(0, t, len(tmp))
#                 plt.plot(t, tmp)
#                 plt.xlabel('time/s')
#                 plt.ylabel('amplitude')
#                 plt.grid()
#                 ax.title.set_text(classes[ys[i]])
#             else:
#                 count = 0
#                 plt.tight_layout()
#                 plt.show()
#                 plt.figure(figsize=(15, 10))
#         break
#
#
# def plot_spectrogram(data):
#     f, t, Sxx = signal.spectrogram(
#         data.reshape(1, -1),
#         fs=300,
#         nperseg=64,
#         noverlap=32
#     )
#     cmap = plt.get_cmap('jet')
#     Sxx = abs(Sxx).squeeze()
#     mask = Sxx > 0
#     Sxx[mask] = np.log(Sxx[mask])
#     plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap=cmap)
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.savefig('./figs/spectrogram.png', bbox_inches='tight', dpi=220)
#     plt.show()
#
#
# def plot_signal(tmp, tmp2, pic_name):
#     t = (len(tmp) - 1) / 300
#     t = np.linspace(0, t, len(tmp))
#     plt.plot(t, tmp, label='origin')
#     plt.plot(t, tmp2, label=pic_name)
#     plt.xlim(10, 12)
#     plt.ylabel('Potential [mV]')
#     plt.xlabel('Time [sec]')
#     plt.legend()
#     plt.savefig(f'./figs/{pic_name}.png', bbox_inches='tight', dpi=220)
#     plt.show()
#
#
# if __name__ == '__main__':
#     data_df = read_data(zip_path='../data/training.zip', data_path='../training')
#     data = data_df.iloc[0, 0] / 1000
#     data = data.reshape(1, -1)
#     dropout = DropoutBursts(2, 10)
#     random = RandomResample()
#     data2 = dropout(data).squeeze()
#     data3 = random(data).squeeze()
#     data = data.squeeze()
#     # plot_spectrogram(data)
#     plot_signal(data, data2, 'DropoutBurst')
#     plot_signal(data, data3, 'RandomResampling')

