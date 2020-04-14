import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def running_mean(x, N):
    divider = np.convolve(np.ones_like(x), np.ones((N,)), mode='same')
    return np.convolve(x, np.ones((N,)), mode='same') / divider


#dir = '../data/data_remote/'
dir = '../data/'
#envs = ['riverswim'] #,,'cartpole', 'mountain', 'riverswim'
envs = ['point_hard']
#settings = ['oac', 'sac', 'sac_no_entropy', 'p-oac', 'p-oac-counts', 'p-oac-5-counts'  ]
#settings = ['oac_', 'sac_', 'g-oac_', 'p-oac_', 'g-oac_2', 'p-oac_2', ]
settings = ['p-oac_']
#settings = ['p-oac-counts'  ]#, ,  'oac','g-oac-ensemble', 'p-oac-ensemble''oac', 'sac',
#settings = [ 'p-oac_',  'g-oac_', 'p-oac_counts', 'g-oac_counts', 'p-oac_means', 'g-oac_means', 'p-oac_means_counts', 'g-oac_means_counts']#, ,  'oac','g-oac-ensemble', 'p-oac-ensemble''oac', 'sac',

# envs = ['cartpole', 'mountain'] #, 'cartpole', 'mountain', 'riverswim'
# settings = ['sac_', 'oac_', 'p-oac_5', 'p-tsac_5', 'g-oac_5', 'g-tsac_5', 'p-oac_', 'p-tsac_', 'g-oac_', 'g-tsac_'] #, 'g-tsac_1'] #, ,  'oac',
colors = ['c', 'k', 'orange', 'purple', 'r', 'b', 'g', 'y', 'brown', 'magenta', '#BC8D0B', "#006400"]
markers = ['o', 's', 'v', 'D', 'x', '*', '|', '+', '^', '2', '1', '3', '4']
fields = ['exploration/Average Returns', 'remote_evaluation/Average Returns',
          'trainer/QF mean', 'trainer/QF std',
          'exploration/Returns Max', 'remote_evaluation/Returns Max',
          'trainer/Policy mu Mean', 'trainer/Policy log std Mean']  # 'trainer/QF std 2']
field_to_label = {
    'remote_evaluation/Average Returns': 'offline return',
    'exploration/Average Returns': 'online return',
    'trainer/QF mean': 'Mean Q',
    'trainer/QF std': 'Std Q',
    'trainer/QF std 2': 'Std Q 2',
    'trainer/QF Unordered': 'Q Unordered samples',
    'trainer/QF target Undordered': 'Q Target Unordered samples',
    'exploration/Returns Max': 'online Max Return',
    'remote_evaluation/Returns Max': 'offline Max Return',
    'trainer/Policy mu Mean': 'policy mu',
    'trainer/Policy log std Mean': 'policy std'
}
separate = True
count = 0
plot_count = 0
n_col = 2
subsample = 1
for env in envs:
    fig, ax = plt.subplots(int(np.ceil(len(fields) / n_col)), n_col, figsize=(12, 24))
    fig.suptitle(env)
    col = 0
    for f, field in enumerate(fields):
        for s, setting in enumerate(settings):
            path = dir + env + '/' + setting + '/*/progress.csv'
            paths = glob.glob(path)
            # print("Path:", path)
            # print("Paths:", paths)
            min_rows = np.inf
            results = []
            final_results = []
            for j, p in enumerate(paths):
                print(p)
                try:
                    data = pd.read_csv(p, usecols=[field])
                    #print(data)
                except:
                    break
                try:
                    res = np.array(data[field], dtype=np.float64)
                except:
                    print("What")
                if separate:
                    if f == 0:
                        label = setting  + '-' + str(j)
                    else:
                        label = None
                    mean = running_mean(res, subsample)
                    x = list(range(len(mean)))
                    ax[col // n_col][col % n_col].plot(x, mean, label=label, color=colors[j])
                    #plt.plot(res, label=setting + '-' + str(j), color=colors[count % len(colors)])
                    count += 1
                else:
                    if len(res) < min_rows:
                        min_rows = len(res)
                    results.append(res)
            if not separate and len(results) > 0:
                print(len(results))
                for i, _ in enumerate(paths):
                    final_results.append(results[i][:min_rows])
                data = np.stack(final_results, axis=0)
                n = data.shape[0]
                #print(data)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                mean = running_mean(mean, subsample)
                std = running_mean(std, subsample)
                x = list(range(len(mean)))
                #indexes = [i * subsample for i in range(len(mean) // subsample)]
                # mean = mean[indexes]
                # std = std[indexes]
                #x = indexes

                if f == 0:
                    label = setting
                else:
                    label = None
                ax[col // n_col][col % n_col].plot(x, mean, label=label, color=colors[s])
                if n > 1:
                    ax[col // n_col][col % n_col].fill_between(x, mean - 2 * (std / np.sqrt(n)),
                                         mean + 2 * (std / np.sqrt(n)),
                                 alpha=0.2, color=colors[s])
        ax[col // n_col][col % n_col].set_title(field_to_label[field], fontdict={'fontsize': 7})
        if field in ['remote_evaluation/Average Returns', 'exploration/Average Returns']:
            ax[col // n_col][col % n_col].set_ylim((-10000, 0))
        if col // n_col == int(np.ceil(len(fields) / n_col)) - 1:
            ax[col // n_col][col % n_col].set_xlabel('epoch', fontdict={'fontsize': 7})
        ax[col // n_col][col % n_col].set_title(field_to_label[field], fontdict={'fontsize': 7})
        col += 1
        plot_count += 1
    fig.legend(loc='lower center', ncol=len(fields)//2)
    plt.show()
# oac = pd.read_csv('oac/1581679951.0551817/progress.csv')
# w_oac = pd.read_csv('../data/riverswim/w-oac/1581679961.4996805/progress.csv')
# sac = pd.read_csv('../data/riverswim/sac/1581692438.1551993/progress.csv')
#
# oac_res = np.array(oac['remote_evaluation/Average Returns'])
# w_oac_res = np.array(w_oac['remote_evaluation/Average Returns'])
# sac_res = np.array(sac['remote_evaluation/Average Returns'])
# plt.plot(oac_res, label='oac')
# plt.plot(w_oac_res, label='w-oac')
# plt.plot(sac_res, label='sac')
