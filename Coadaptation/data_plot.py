# Import Libraries and define functions
import csv
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
plt.rcParams.update({'font.size': 22})

def set_style():
    plt.minorticks_on()
    plt.grid(which='minor', color='white', linestyle='-', alpha=0.4)
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 22})
    legend = plt.gca().legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., frameon = 1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('white')


def save_fig(file_name):
    plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.02,
        frameon=None, metadata=None)


def read_file(path):
  with open(path, 'r') as fd:
    cwriter = csv.reader(fd)
    data = []
    skipped_first_row = False
    for row in cwriter:
      if not skipped_first_row:
        skipped_first_row = True
        continue
      row_tmp = row
      for idx in range(len(row_tmp)):
        row_tmp[idx] = float(row_tmp[idx])
      data.append(row_tmp)
    config = data[0]
    data = data[1]
  return data, config


def read_data_folder(path):
  data_max = []
  data_all = []
  configs = []
  for idx in range(1, DESIGNS+1):
    data, config = read_file(path + '{}.csv'.format(idx))
    for d in data:
        data_all.append(d)
    data_max.append(np.amax(data))
    data = np.array(data)
    data_all.append(data)
    configs.append(config)
  return data_all, data_max, configs
    

def compute_mean_std(paths):
  data_max_ar = []
  data_top_mean_ar = []
  configs_ar = []
  data_all_ar = []
  for p in paths:
    data_all, data_max, configs = read_data_folder(p)
    data_all_ar.append(data_all)
    data_max_ar.append(data_max)
    configs_ar.append(configs)
  data_max_mean = np.mean(data_max_ar, axis=0)
  data_max_std = np.std(data_max_ar, axis=0)
  return data_max_mean, data_max_std


def plot_data_means(mean_data, std_data, color, label='', plot_random_std=False):
  x = np.arange(1, len(mean_data) + 1)
  init_data_mean = mean_data[0:INITIAL_DESIGNS+1]
  init_data_std = std_data[0:INITIAL_DESIGNS+1]
  init_x = x[0:INITIAL_DESIGNS+1]
  optim_data_mean = mean_data[INITIAL_DESIGNS::2]
  optim_data_std = std_data[INITIAL_DESIGNS::2]
  optim_x = x[INITIAL_DESIGNS::2]
  random_data_mean = mean_data[INITIAL_DESIGNS+1::2]
  random_data_std = std_data[INITIAL_DESIGNS+1::2]
  random_x = x[INITIAL_DESIGNS+1::2]
  plt.plot(init_x, init_data_mean, color=color, alpha=0.3, linewidth=2.0)
  plt.fill_between(init_x, init_data_mean - init_data_std, init_data_mean + init_data_std, facecolor=color, alpha=0.1)
  plt.plot(optim_x, optim_data_mean, color=color, label=label, linewidth=2.0)
  plt.fill_between(optim_x, optim_data_mean - optim_data_std, optim_data_mean + optim_data_std, facecolor=color, alpha=0.2)
  #plt.plot(random_x, random_data_mean, color=color, linestyle='--')
  if plot_random_std:
    plt.fill_between(random_x, random_data_mean - random_data_std, random_data_mean + random_data_std, facecolor=color, alpha=0.2)
  set_style()

    
def plot_data_means_optim_random(mean_data, std_data, color, label=''):
  x = np.arange(1, len(mean_data) + 1)
  init_data_mean = mean_data[0:INITIAL_DESIGNS+1]
  init_data_std = std_data[0:INITIAL_DESIGNS+1]
  init_x = x[0:INITIAL_DESIGNS+1]
  optim_data_mean = mean_data[INITIAL_DESIGNS::2]
  optim_data_std = std_data[INITIAL_DESIGNS::2]
  optim_x = x[INITIAL_DESIGNS::2]
  random_data_mean = mean_data[INITIAL_DESIGNS+1::2]
  random_data_std = std_data[INITIAL_DESIGNS+1::2]
  random_x = x[INITIAL_DESIGNS+1::2]
  plt.plot(init_x, init_data_mean, color=color, alpha=0.3, linewidth=2.0)
  plt.fill_between(init_x, init_data_mean - init_data_std, init_data_mean + init_data_std, facecolor=color, alpha=0.1)
  plt.plot(optim_x, optim_data_mean, color=color, linewidth=2.0, label=label)
  plt.fill_between(optim_x, optim_data_mean - optim_data_std, optim_data_mean + optim_data_std, facecolor=color, alpha=0.2)
  plt.plot(random_x, random_data_mean, color=color, label=label + ' Random Exploration', linestyle='--', linewidth=2.0)
  plt.fill_between(random_x, random_data_mean - random_data_std, random_data_mean + random_data_std, facecolor=color, alpha=0.2)
  set_style()


# Change these parameters as required
INITIAL_DESIGNS = 5
DESIGNS = 55

# Proposed method using data-driven design optimization
# Change folder names
# EXPERIMENT_FOLDERS = [
#     'data_exp_sac_pso_batch/Fri_Mar_15_16:03:45_2024__a8e9d6f2',
#     'data_exp_sac_pso_batch/Mon_Mar_18_17:15:50_2024__72feed47',
#     'data_exp_sac_pso_batch/Thu_Mar_21_12:51:32_2024__c1b9b1d3',
#     'data_exp_sac_pso_batch/Fri_Mar_22_13:30:55_2024__2923c607',
#     'data_exp_sac_pso_batch/Mon_Mar_25_10:06:11_2024__bd0c4004',
#     # 'data_exp_sac_pso_batch/Sun_Oct_20_23:36:49_2019__5db36c30',
#     # 'data_exp_sac_pso_batch/Sun_Oct_20_23:37:06_2019__30302df7',
# ]

EXPERIMENT_FOLDERS = [
    # 'data_exp_sac_pso_batch/Wed_Mar_27_10:00:50_2024__85e11d18',
    'data_exp_sac_pso_batch/Thu_May__9_00:58:25_2024__7db9fd18'
    # 'data_exp_sac_pso_batch/Sun_Oct_20_23:36:49_2019__5db36c30',
    # 'data_exp_sac_pso_batch/Sun_Oct_20_23:37:06_2019__30302df7',
]



# Baseline using simulations for the evaluation of design candidates
# Change folder names
EXPERIMENTS_FOLDERS_2 = [
    # 'data_exp_sac_pso_sim/Wed_Oct_23_05:49:05_2019__32509883',
    # 'data_exp_sac_pso_sim/Wed_Oct_23_05:49:22_2019__6878474d',
    # 'data_exp_sac_pso_sim/Wed_Oct_23_05:51:21_2019__36c0fafc',
    # 'data_exp_sac_pso_sim/Wed_Oct_23_05:51:33_2019__b21f861b',
    # 'data_exp_sac_pso_sim/Wed_Oct_23_05:51:45_2019__cc74a513',
]

exp_files = ['{}/data_design_'.format(folder) for folder in EXPERIMENT_FOLDERS] # Novelty Search + PSO on Q
exp_mean, exp_std = compute_mean_std(exp_files)

# Plot performance of optimized designs and randomly selected designs
plot_data_means_optim_random(exp_mean, exp_std, color='red', label='Proposed Method')
plt.ylabel('Cum. Episodic Reward')
plt.xlabel('Designs')
# plt.ylim([0,200])
plt.xlim([1,DESIGNS])
#save_fig('plots_HalfCheetah_random_vs_novelty_search.pdf')
plt.savefig(f'evaluation/plots_Snake_random_vs_novelty_search.png',bbox_inches="tight")
plt.savefig(f'evaluation/plots_Snake_random_vs_novelty_search.pdf',bbox_inches="tight")

# Compare two experiments/methods against each other
exp_files = ['{}/data_design_'.format(folder) for folder in EXPERIMENT_FOLDERS] # Novelty Search + PSO on Q
exp_mean, exp_std = compute_mean_std(exp_files)
plot_data_means(exp_mean, exp_std, color='red', label='Proposed Method')

# exp_files = ['{}/data_design_'.format(folder) for folder in EXPERIMENTS_FOLDERS_2] # Novelty Search + PSO on Q
# exp_mean, exp_std = compute_mean_std(exp_files)
# plot_data_means(exp_mean, exp_std, color='blue', label='Using Simulations')
plt.ylabel('Cum. Episodic Reward')
plt.xlabel('Designs')
# plt.ylim([0,200])
plt.xlim([1,DESIGNS])
plt.savefig(f'evaluation/cumulative_reward.png',bbox_inches="tight")
plt.savefig(f'evaluation/cumulative_reward.pdf',bbox_inches="tight")