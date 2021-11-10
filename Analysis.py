import matplotlib.pyplot as plt

# Actions frequency means the number of steps between two consecutive steps.
actions_frequency = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
# Mean episode reward that based on different action frequency
rewards = {
    'ShortQPolicy': (-1566.65, -2204.58, -2745.91, -3306.14, -3322.42, -3575.44, -4100.49, -3918.7, -4464.57, -4131.26, -4417.94),
    'RandomPolicy': (-799.68, -1607.42, -2427.05, -2983.97, -3317.99, -3553.98, -3767.18, -3988.17, -4314.39, -4132.04, -4278.20),
    'ppo_256_tanh': (0, -26292.00),
    'belief_ppo_256_tanh': (0, -25683.33),
    'belief_ppo_128_relu': (-10011.67, )
}

episode_len = {
    'ShortQPolicy': (60.56, 61.7, 61.98, 62.2, 62.54, 62.7, 64.83, 65.67, 65.0, 63.42, 67.91),
    'RandomPolicy': (60.73, 61.18, 61.77, 62.13, 62.58, 62.65, 64.71, 65.67, 65.00, 63.33, 67.91),
    'belief_ppo_256_tanh': (0, 67.15)
}

episode_pkg_drop = {
    'ShortQPolicy': (0.31, 0.44, 0.55, 0.66, 0.66, 0.72, 0.82, 0.78, 0.89, 0.83, 0.88),
    'RandomPolicy': (0.16, 0.32, 0.49, 0.6, 0.66, 0.71, 0.75, 0.8, 0.86, 0.83, 0.85),
    'belief_ppo_256_tanh': (0, 67.15)
}


if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    plt.plot(actions_frequency, rewards['ShortQPolicy'])
    plt.plot(actions_frequency, rewards['RandomPolicy'])
    plt.xlabel('actions_frequency')
    plt.ylabel('Mean episode reward')
    plt.legend(['ShortQPolicy', 'RandomPolicy'])
    plt.title('Mean episode reward vs. No. of actions_frequency')

    plt.figure(figsize=(10, 5))
    plt.plot(actions_frequency, episode_len['ShortQPolicy'])
    plt.plot(actions_frequency, episode_len['RandomPolicy'])
    plt.xlabel('actions_frequency')
    plt.ylabel('Mean episode length')
    plt.legend(['ShortQPolicy', 'RandomPolicy'])
    plt.title('Mean episode length vs. No. of actions_frequency')

    plt.figure(figsize=(10, 5))
    plt.plot(actions_frequency, episode_pkg_drop['ShortQPolicy'])
    plt.plot(actions_frequency, episode_pkg_drop['RandomPolicy'])
    plt.xlabel('actions_frequency')
    plt.ylabel('package drop rate')
    plt.legend(['ShortQPolicy', 'RandomPolicy'])
    plt.title('Package drop rate vs. No. of actions_frequency')
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

