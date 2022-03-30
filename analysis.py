import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def figure_configuration_ieee_standard():
    # IEEE Standard Figure Configuration - Version 1.0
    # run this code before the plot command
    # According to the standard of IEEE Transactions and Journals:
    # Times New Roman is the suggested font in labels.
    # For a single part figure, labels should be in 8 to 10 points,
    # whereas for a multipart figure, labels should be in 8 points.
    # Width: column width: 8.8 cm; page width: 18.1 cm.
    # width & height of the figure
    k_scaling = 1
    # scaling factor of the figure
    # (You need to plot a figure which has a width of (8.8 * k_scaling)
    # in MATLAB, so that when you paste it into your paper, the width will be
    # scaled down to 8.8 cm  which can guarantee a preferred clearness.
    k_width_height = 1.5  #1.3  # width:height ratio of the figure

    # fig_width = 17.6/2.54 * k_scaling
    fig_width = 8.8/2.54 * k_scaling
    fig_height = fig_width / k_width_height

    # ## figure margins
    # top = 0.5  # normalized top margin
    # bottom = 3	# normalized bottom margin
    # left = 4	# normalized left margin
    # right = 1.5  # normalized right margin

    params = {"axes.labelsize": 12,  # fontsize for x and y labels (was 10)
              "axes.titlesize": 10,
              "font.size": 8,  # was 10
              "legend.fontsize": 8,  # was 10
              "xtick.labelsize": 10,
              "ytick.labelsize": 12,
              "figure.figsize": [fig_width, fig_height],
              "font.family": "serif",
              "font.serif": ["Times New Roman"],
              "lines.linewidth": 2.5,
              "axes.linewidth": 1,
              "axes.grid": True,
              "savefig.format": "pdf",
              "axes.xmargin": 0,
              "axes.ymargin": 0,
              "savefig.pad_inches": 0.04,
              "legend.markerscale": 0.9,
              "savefig.bbox": "tight",
              "lines.markersize": 2,
              "legend.numpoints": 4,
              "legend.handlelength": 2.0, #was 3.5
              "text.usetex": True
              }

    matplotlib.rcParams.update(params)


# Actions frequency means the number of steps between two consecutive steps.
syncT = (1, 2, 3, 4, 5, 6)
# Mean episode packet drop rate that based on different action frequency
syncT_drop_rate = {
    "name": "episode_pkg_drop",
    "JSQ": (0.04870370370370371, 0.07694444444444444, 0.1262037037037037, 0.18675925925925926, 0.24888888888888888, 0.31379629629629624),
    "ATVC": (0.055185185185185184, 0.08453703703703705, 0.09837962962962962, 0.12148148148148148, 0.16009259259259256, 0.1885185185185185),
    "BicNet": (0.07541666666666667, 0.0837962962962963, 0.09972222222222223, 0.1288888888888889, 0.1768981481481481, 0.2159722222222222),
    "CommNet": (0.07680555555555554, 0.09046296296296297, 0.10828703703703703, 0.13611111111111113, 0.17004629629629625, 0.2113425925925926),
    "IPPO_DO": (0.09485185185185185, 0.09523148148148147, 0.10236111111111111, 0.13069444444444445, 0.15458333333333332, 0.17884259259259258),
    "IPPO_TO": (0.06828703703703703, 0.06439814814814815, 0.08935185185185185, 0.12111111111111111, 0.16291666666666665, 0.19606481481481483),
}

syncT_std = {
    "JSQ": ((0.04160319432074347, 0.055804213086663945), (0.07030028306752989, 0.08358860582135899), (0.11950045448676945, 0.13290695292063798), (0.17925101181008563, 0.1942675067084329), (0.24132731361751497, 0.2564504641602628), (0.30596671136509146, 0.321625881227501)),
    "VC": ((0.0491218526228569, 0.06115592515492088), (0.06784776572536663, 0.08141149353389263), (0.08791277862636446, 0.10153166581807996), (0.11496044550914043, 0.13013214708345214), (0.14754238466668268, 0.16366131903702102), (0.18285188522627174, 0.19835181847743197)),
    "ATVC": ((0.048560076472870126, 0.06181029389750024), (0.07775711020418179, 0.09131696386989231), (0.09112018341649476, 0.10563907584276448), (0.11472691258568508, 0.12823605037727787), (0.15252946718085603, 0.1676557180043291), (0.18113942821245366, 0.19589760882458335)),
    "BicNet": ((0.06842205688503576, 0.08241127644829759), (0.07568015727105454, 0.09191243532153806), (0.09165502609732737, 0.10778941834711708), (0.12163037174353808, 0.1361474060342397), (0.16891952345816966, 0.18487677283812656), (0.2082608779195033, 0.22368356652494112)),
    "CommNet": ((0.06941507126076943, 0.08419603985034166), (0.08341582323554712, 0.09751010269037881), (0.10049650005507031, 0.11607757401900375), (0.1288392967627169, 0.14338292545950534), (0.16170898467922365, 0.17838360791336885), (0.20370541657773927, 0.21897976860744595)),
    "IPPO_DO": ((0.09165569863862957, 0.09804800506507413), (0.08774571387259397, 0.10271724909036897), (0.09452754509628428, 0.11019467712593795), (0.12227352762712924, 0.13911536126175963), (0.14626294541313775, 0.1629037212535289), (0.17145367554222007, 0.1862315096429651)),
    "IPPO_TO": ((0.06021086886921531, 0.07636320520485876), (0.05735905478882176, 0.07143724150747455), (0.08169562826402282, 0.09700807543968087), (0.1134670540254497, 0.12875516819677255), (0.1551313889726285, 0.17070194436070482), (0.18812195747427835, 0.20400767215535132)),
    "gg_ATVC": ((0.16370409423803114, 0.1686025724286355), (0.1992967081684404, 0.20394329183155954), (0.2673013066437868, 0.27183202668954654), (0.3601322062756319, 0.36418112705770145), (0.4501167358444095, 0.4539565974889238))
}

n_agents = (3, 9, 27, 54)
# Mean episode packet drop rate that based on different action frequency
Nagents_drop_rate = {
    "name": "episode_pkg_drop",
    "JSQ": (0.046625, 0.04718055555555555, 0.1171957671957672, 0.22079475308641974),
    "ATVC": (0.055185185185185184, 0.06842592592592592, 0.13575837742504407, 0.23005401234567902),
    "BicNet": (0.07541666666666667, 0.07106481481481479, 0.134347442680776, 0.2455632716049383),
    "CommNet": (0.07680555555555554, 0.07754629629629628, 0.14193121693121694, 0.239429012345679 ),
    "IPPO_DO": (0.09485185185185185, 0.09155555555555556,0.1471516754850088 ,0.24453703703703705 ),
    "IPPO_TO": (0.06828703703703703, 0.06643518518518518, 0.12687830687830687, 0.22432870370370372 )
}

Nagents_std = {
    "JSQ": ((0.044663749793763245, 0.048586250206236754), (0.04513433733346386, 0.04922677377764725), (0.11090209815088321, 0.12348943624065119), (0.21610985783863787, 0.22547964833420162)),
    "ATVC": ((0.048560076472870126, 0.06181029389750024),(0.060878503375441304, 0.07597334847641053) , (0.12971918433020097, 0.14179757051988717), (0.22595751280744997, 0.23415051188390806)),
    "BicNet": ((0.06842205688503576, 0.08241127644829759), (0.06455042692538535, 0.07757920270424423), (0.12837376985987656, 0.14032111550167545),(0.2400608591900825, 0.2510656840197941)),
    "CommNet": ((0.06941507126076943, 0.08419603985034166), (0.07101956202760736, 0.0840730305649852), (0.13570506568170196, 0.1481573681807319),(0.23479160439557373, 0.2440664202957843) ),
    "IPPO_DO": ((0.09165569863862957, 0.09804800506507413), (0.08841728865801121, 0.0946938224530999) , (0.14424293325461593, 0.15006041771540166) , (0.24244568639442032, 0.24662838767965378)),
    "IPPO_TO": ((0.06021086886921531, 0.07636320520485876), (0.06431375935704798, 0.06855661101332238), (0.12413344793076107, 0.12962316582585265),(0.22222330952847774, 0.2264340978789297))
}


heat_map = {
    "JSQ": [[0.47729673, 1., 1., 1., 1., 1.],
            [0., 0.47680412, 1., 1., 1., 1.],
            [0., 0., 0.49046322, 1., 1., 1.],
            [0., 0., 0., 0.50326797, 1., 1.],
            [0., 0., 0., 0., 0.5245098, 1.],
            [0., 0., 0., 0., 0., 0.5]],
    "ATVC":[[0.85564784 ,0.82230997, 0.92314514, 0.94098088, 0.97430957 ,0.97727273],
             [0.71343284 ,0.70849637, 0.86320755, 0.9147929 , 0.93607306, 0.97931034],
             [0.26443203, 0.3010003 , 0.73859987, 0.84795539, 0.90575916, 0.97632653],
             [0.06640793 ,0.07425541, 0.33716191 ,0.63857311 ,0.78567775 ,0.94760148],
             [0.02110626, 0.00953984 ,0.08895552 ,0.2202729  ,0.41781198, 0.83644189],
             [0.00590319 ,0.00189934 ,0.01594203 ,0.05198973 ,0.14718019, 0.4972592 ]],
    "IPPO_TO": [[0.66777205, 0.73203832, 0.80211574, 0.85448246, 0.91848617, 0.92804878],
                [0.63873518, 0.68033257, 0.77940718, 0.84122329, 0.88895743, 0.93419633],
                [0.54060032, 0.61370815, 0.70326797, 0.78896605, 0.86697483, 0.90729614],
                [0.469163, 0.54524451, 0.63741524, 0.74307992, 0.83391696, 0.88380537],
                [0.3613963, 0.44101732, 0.56551388, 0.65692381, 0.78003291, 0.85017921],
                [0.27284595, 0.36116505, 0.50336072, 0.6417704, 0.74374177, 0.82200886]],
    "IPPO_DO": [[0.74830422, 0.76662392, 0.78665948, 0.80893643, 0.82187078, 0.84082777],
             [0.68427139 ,0.70131427 ,0.72296738 ,0.75316697, 0.78401095, 0.80081321],
             [0.59585517, 0.62073846, 0.65822022 ,0.67257768, 0.71914856, 0.72195229],
             [0.51170525 ,0.53461289, 0.57377004 ,0.60393031 ,0.64216299, 0.64763669],
             [0.44494683 ,0.46005253, 0.48925345 ,0.52759681 ,0.57111159, 0.58855111],
             [0.36648319 ,0.372148  , 0.41500361 ,0.43678528 ,0.45490884, 0.50673442]],
    "CommNet": [[0.35434783 ,0.4625032 , 0.50684532, 0.7254607 , 0.84735707, 0.89753773],
             [0.26811405 ,0.46267335 ,0.53034729, 0.68273245, 0.8144276,  0.88849178],
             [0.16703297 ,0.321843  , 0.5035791,  0.64059366 ,0.76158599 ,0.84507042],
             [0.12407862 ,0.2083499 , 0.4691095,  0.62735426 ,0.74847982, 0.84397163],
             [0.09052632 ,0.18077803 ,0.44984326 ,0.60809313, 0.69677419 ,0.79657054],
             [0.09146341, 0.17733231 ,0.40941011, 0.59570042, 0.7244898 , 0.74664107]],
    "BicNet": [[0.80846634 ,0.84136986 ,0.94387587, 0.97630148 ,0.99380485, 0.99291785],
               [0.63020429 ,0.72694394 ,0.82785496, 0.89988709 ,0.94008559 ,0.96161228],
               [0.24848485, 0.416833  , 0.67567568 ,0.78421247 ,0.89767238 ,0.93725043],
               [0.17906206 ,0.270223  , 0.51292921 ,0.59732105 ,0.68188302 ,0.8481153 ],
               [0.05828221 ,0.12093628 ,0.32140892, 0.54268624, 0.54986807 ,0.57107387],
               [0.01317365, 0.04568024, 0.10487805 ,0.42268041 ,0.52472325, 0.58168903]],
}


def plot(dic, std_dic, x, x_label=r"$\Delta T$", y_label=r"Average packet drop rate$(\%)$"):
    plt.figure()
    # Plot mean of rewards.
    for k, v in dic.items():
        if k == "name":
            continue
        plt.plot(x, np.array(v)*100, label=k)
        # Plot confidence interval
        r_low, r_high = [], []
        for xx, yy in std_dic[k]:
            r_low.append(xx*100)
            r_high.append(yy*100)
        plt.fill_between(x, r_low, r_high, alpha=0.1)

    plt.xlabel(x_label)
    plt.grid()
    plt.ylabel(y_label)
    plt.legend(ncol=2, columnspacing=1, borderpad=0.2, labelspacing=0.2)


def smooth(data, weight=0.95):
    scalar = data.values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    res = pd.Series(smoothed, name=data.name)
    return res


def plot_learning_curve():
    res = []
    models = ("ATVC", "BicNet", "CommNet", "IPPO_TO", "IPPO_DO")
    for model in models:
        df = pd.read_csv("./Data/" + model + ".csv")
        df_r = df.loc[:, "ray/tune/episode_reward_mean"]
        df_r = smooth(df_r)
        if model == "IPPO_TO":
            res.append(df_r.rename("IPPO\_TO"))
        elif model == "IPPO_DO":
            res.append(df_r.rename("IPPO\_DO"))
        else:
            res.append(df_r.rename(model))
    total = pd.concat(res, axis=1)
    total.plot()
    plt.xlabel("Iteration")
    plt.ylabel("Mean episode reward")
    plt.legend(ncol=2, columnspacing=1, borderpad=0.2, labelspacing=0.2)
    plt.savefig("../paper_figure/reward_comparison.pdf", dpi=300)
    plt.show()


def plot_communication_rate():
    # in matplotlib fontsize in pixel, 1 pixel = 4/3 pt
    res = []
    data = ("comm_reduction1", "comm_reduction2")
    for model in data:
        df = pd.read_csv("./Data/" + model + ".csv")
        df_r = df.loc[:, "comm_number"]
        res.append(df_r.rename(model))
    total = pd.concat(res, axis=1)
    v = total.mean(axis=1).values
    new_v = []
    for i in range(120):
        s = 0
        for j in range(83):
            s += v[i*83+j]
        new_v.append(s/83)
    total = pd.Series(new_v, name="ATVC")
    ones = [1]*len(total)
    ones_ser = pd.Series(ones, name="ATVC without attention")
    total = pd.concat([total, ones_ser], axis=1)
    total.plot()
    plt.xlabel("Episode")
    plt.ylabel("Mean communication rate")
    plt.legend(ncol=2, columnspacing=1, borderpad=0.2, labelspacing=0.2)
    plt.savefig("../paper_figure/communication_rate.pdf", dpi=300)
    plt.show()


def plot_drop_rate_syncT():
    plot(syncT_drop_rate, syncT_std, syncT)
    plt.savefig("../paper_figure/deltaT.pdf", dpi=300)
    plt.show()


def plot_drop_rate_Nagents():
    plot(Nagents_drop_rate, Nagents_std, n_agents)
    plt.savefig("../paper_figure/deltaT.pdf", dpi=300)
    plt.show()
    

def plot_scheduling_heatmap():
    for k, v in heat_map.items():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim(-0.5, 5.5)
        a = np.asarray(v)
        im = ax.imshow(a, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        cb = fig.colorbar(im)
        cb.ax.tick_params()
        plt.xlabel(r"$b_1$")
        plt.ylabel(r"$b_2$")
        plt.savefig("../paper_figure/" + k + "_heatmap.pdf", dpi=300)
        plt.show()


if __name__ == "__main__":
    figure_configuration_ieee_standard()
    plot_learning_curve()
    plot_communication_rate()
    plot_drop_rate_syncT()
    plot_drop_rate_Nagents()
    plot_scheduling_heatmap()

