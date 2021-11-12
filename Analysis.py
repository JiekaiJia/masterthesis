import matplotlib.pyplot as plt

# Actions frequency means the number of steps between two consecutive steps.
actions_frequency = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
# Mean episode reward that based on different action frequency
rewards = {
    'ShortQPolicy': (-76.31, -98.03, -131.91, -179.79, -202.16, -214.82, -224.43, -232.39, -236.78, -245.22, -255.93),
    'RandomPolicy': (-90.99, -113.58, -140.9, -167.35, -196.23, -222.81, -248.45, -271.7, -297.7, -315.72, -336.48),
    'ppo_256_tanh': (-66.2, -70.92, -81.48, -99.94, -111.65, -132.74, -152.82, -176.44, -200.39, -223.0, -243.7),
    'belief_ppo_256_tanh': (),
    'belief_ppo_128_relu': ()
}

episode_len = {
    'ShortQPolicy': (81.33, 82.3, 81.85, 82.4, 82.92, 82.8, 83.84, 83.07, 83.97, 85.67, 86.61),
    'RandomPolicy': (80.58, 81.68, 82.15, 82.57, 82.46, 83.8, 84.08, 83.87, 84.58, 84.75, 85.88),
    'ppo_256_tanh': (60.67, 61.43, 61.95, 62.33, 62.50, 63.10, 64.77, 66.00, 65.00, 62.83, 68.00),
}

episode_pkg_drop = {
    'ShortQPolicy': (0.13, 0.16, 0.22, 0.3, 0.34, 0.36, 0.37, 0.39, 0.39, 0.41, 0.43),
    'RandomPolicy': (0.15, 0.19, 0.24, 0.28, 0.33, 0.37, 0.41, 0.45, 0.5, 0.53, 0.56),
    'ppo_256_tanh': (0.15, 0.31, 0.49, 0.59, 0.66, 0.71, 0.75, 0.78, 0.83, 0.84, 0.83),
}


if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    plt.plot(actions_frequency, rewards['ShortQPolicy'])
    plt.plot(actions_frequency, rewards['RandomPolicy'])
    plt.plot(actions_frequency, rewards['ppo_256_tanh'])
    # plt.xlim((0, 3))
    # plt.ylim((-3000, -600))
    plt.xlabel('actions_frequency')
    plt.ylabel('Mean episode reward')
    plt.legend(['ShortQPolicy', 'RandomPolicy', 'ppo_256_tanh'])
    plt.title('Mean episode reward vs. No. of actions_frequency')

    # plt.figure(figsize=(10, 5))
    # plt.plot(actions_frequency, episode_len['ShortQPolicy'])
    # plt.plot(actions_frequency, episode_len['RandomPolicy'])
    # plt.plot(actions_frequency, episode_len['ppo_256_tanh'])
    # plt.xlabel('actions_frequency')
    # plt.ylabel('Mean episode length')
    # plt.legend(['ShortQPolicy', 'RandomPolicy', 'ppo_256_tanh'])
    # plt.title('Mean episode length vs. No. of actions_frequency')
    #
    plt.figure(figsize=(10, 5))
    plt.plot(actions_frequency, episode_pkg_drop['ShortQPolicy'])
    plt.plot(actions_frequency, episode_pkg_drop['RandomPolicy'])
    plt.plot(actions_frequency, episode_pkg_drop['ppo_256_tanh'])
    plt.xlabel('actions_frequency')
    plt.ylabel('package drop rate')
    plt.legend(['ShortQPolicy', 'RandomPolicy', 'ppo_256_tanh'])
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

