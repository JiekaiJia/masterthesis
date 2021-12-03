import matplotlib.pyplot as plt
import pandas as pd

# Actions frequency means the number of steps between two consecutive steps.
actions_frequency = (0, 1, 2, 3, 4, 5)
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
    'ShortQPolicy': (0.056527777777777774, 0.15243055555555554, 0.197625, 0.22984722222222223, 0.2538888888888889, 0.27381944444444445),
    # 'RandomPolicy1': (0.1498611111111111, 0.18772222222222223, 0.23518055555555553, 0.28177777777777785, 0.32590277777777776, 0.37120833333333336),
    # 'RandomPolicy2': (0.1122638888888889, 0.12443055555555556, 0.13470833333333335, 0.1569861111111111, 0.18775, 0.21491666666666667),
    'ppo_256_tanh': (0.07912499999999999, 0.09329166666666668, 0.11815277777777779, 0.14541666666666667, 0.1818611111111111, 0.22575000000000003),
    'belief_ppo_256_tanh': (0.10352777777777776, 0.11155555555555556, 0.1284722222222222, 0.1454861111111111, 0.16969444444444445, 0.20640277777777777),
}

std_episode_pkg_drop = {
    'ShortQPolicy': ((0.05322539731216354, 0.05983015824339201), (0.14797272169056636, 0.15688838942054473), (0.19277645439041538, 0.2024735456095846), (0.22454309114286103, 0.23515135330158343), (0.24839402195126006, 0.2593837558265177), (0.2685733773136266, 0.27906551157526227)),
    # 'RandomPolicy1': ((0.14534908305328847, 0.15437313916893375), (0.18320101655052812, 0.19224342789391635), (0.23006053665191548, 0.2403005744591956), (0.27568646662640656, 0.28786908892914914), (0.3202601929953003, 0.33154536256025524), (0.3660766838724122, 0.3763399827942545)),
    # 'RandomPolicy2': ((0.10815098542799922, 0.11637679234977857), (0.11957417742995066, 0.12928693368116045), (0.13049813859210693, 0.13891852807455976), (0.1521147110694248, 0.1618575111527974), (0.18281183640027093, 0.19268816359972907), (0.21057392891422894, 0.2192594044191044)),
    'ppo_256_tanh': ((0.07524881917390742, 0.08300118082609255), (0.08897877974503102, 0.09760455358830233), (0.11357459967570173, 0.12273095587985385), (0.14084674989435716, 0.14998658343897617), (0.17783926587876053, 0.1858829563434617), (0.2211848254174243, 0.23031517458257578)),
    'belief_ppo_256_tanh': ((0.0993334599320441, 0.10772209562351143), (0.10720880083526364, 0.11590231027584748), (0.1238107096571997, 0.13313373478724472), (0.14121196060314012, 0.14976026161908207), (0.1654637712977645, 0.17392511759112442), (0.20204336231105763, 0.2107621932444979)),
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

    plt.xlabel('delta t')
    plt.ylabel(f"Mean episode {dic['name']}")
    plt.legend(labels)
    plt.title(f"Mean episode {dic['name']} vs. No. of actions_frequency")


if __name__ == '__main__':
    # plot(rewards, std_rewards)
    # plot(episode_pkg_drop, std_episode_pkg_drop)
    # plot(comm, std_comm)
    # plt.show()
    loss = pd.read_csv('./runs/encoder_loss.csv')
    loss.plot(x='Step', y='Value')
    plt.ylabel("Loss")
    plt.show()

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

