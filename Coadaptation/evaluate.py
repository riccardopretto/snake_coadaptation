#plot designs

import numpy as np
import sys
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from natsort import natsorted
import torch
import gymnasium as gym
import sys
sys.path.insert(0, '/home/riccardo/Desktop/kevin_exp/Coadaptation/Environments')
import snake_v14
from coadapt import Coadaptation as Coadaptation

design_folder_path = "data_exp_sac_pso_batch"

design_path = sys.argv[1]

design_folder_path = f"{design_folder_path}/{design_path}"
checkpoint_folder_path = f"{design_folder_path}/checkpoints"

save_dir=(f"{design_folder_path}/evaluation")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def read_second_row_as_int(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        if len(rows) >= 2:
            # return [int(cell) for cell in rows[1]]
            return [int(float(cell)*100) for cell in rows[1]]
    

def calculate_average_last_10_values(row):
    row = np.sort(row)
    last_10_values = row[-10:]
    last_10_values = [float(value) for value in last_10_values if value.strip()]  # Convert to integers, ignoring empty strings
    if last_10_values:
        return sum(last_10_values) / len(last_10_values)
    else:
        return None
    



def read_average_last_10_values(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        if len(rows) >= 3:
            last_row = rows[2]
            return calculate_average_last_10_values(last_row)

def read_csv_files_in_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.startswith("data_design_") and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            second_row = read_second_row_as_int(file_path)
            if second_row:
                average_last_10_values = read_average_last_10_values(file_path)
                # Extract number from filename
                file_number = int(filename.split("_")[2].split(".")[0])
                data.append((file_number, filename, second_row, average_last_10_values))
    # Sort data based on the number in filename
    data.sort(key=lambda x: x[0])
    return [(filename, second_row, average_last_10_values) for _, filename, second_row, average_last_10_values in data]

# def read_chk_files_in_folder(folder_path):
#     data = []
#     for filename in os.listdir(folder_path):
#         if filename.startswith(f"checkpoint_design") and filename.endswith(".chk"):
#             file_path = os.path.join(folder_path, filename)
#             data.append(file_path)
#             data_sorted = natsorted(data)
#     return data_sorted

def discard_random_from_list(list):
    list_optimized = []
    for idx in range(len(list)):
        if idx > 4 and idx%2 == 1:      #in case it s the other one, just change ==0 to ==1, (==1 should be the optimized)
            list_optimized.append(list[idx])
    return list_optimized

def discard_optimized_from_list(list):
    list_optimized = []
    for idx in range(len(list)):
        if idx > 4 and idx%2 == 0:      #in case it s the other one, just change ==0 to ==1, (==1 should be the optimized)
            list_optimized.append(list[idx])
    return list_optimized

def order_list_by_reward(list):
    # sorted_list = sorted(list, key=lambda x: x[2], reverse=True)
    sorted_list = sorted(list, key=lambda x: x[2], reverse=False)
    return sorted_list

designs_list = read_csv_files_in_folder(design_folder_path)
# trained_model_list = read_chk_files_in_folder(checkpoint_folder_path)

optimized_design_list = discard_random_from_list(designs_list)
random_design_list = discard_optimized_from_list(designs_list)
# trained_model_list = discard_random_from_list(trained_model_list)

sorted_optimized_list = order_list_by_reward(optimized_design_list)
sorted_random_list = order_list_by_reward(random_design_list)



## Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each design as separate bars with different colors
num_designs = len(optimized_design_list)
num_segments = len(optimized_design_list[0][1])
bar_width = 0.8 / num_designs

# Create a list of colors, one for each design
colors = plt.cm.get_cmap('turbo', num_designs)

for i, (filename, values, _) in enumerate(optimized_design_list):
    x_values = np.arange(num_segments) + i * bar_width
    color = colors(i)  # Select color for this design
    ax.bar(x_values, values, width=bar_width, color=color, label=filename)

# Set labels and legend
ax.set_xlabel('Segment')
ax.set_ylabel('Value')
ax.set_title('Segmented Heatmap Bar Plot')
ax.set_xticks(np.arange(num_segments) + 0.4)
ax.set_xticklabels([f'Segment {i+1}' for i in range(num_segments)])
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


plt.savefig(f'{save_dir}/segment.png',bbox_inches="tight")
plt.savefig(f'{save_dir}/segment.pdf',bbox_inches="tight")



color=["purple","blue", "cyan", "green", "yellow", "orange", "pink", "red"]
# Extracting data
design_names = [design[0] for design in optimized_design_list]
design_values = [design[1] for design in optimized_design_list]

bottom = np.zeros(len(optimized_design_list))

label_list=["1st segment", "2nd segment", "3rd segment", "4th segment", "5th segment", "6th segment", "7th segment", "8th segment"]


# Plotting
fig, ax = plt.subplots(figsize=(10, 6))


def transform_and_transpose(data_list):
    # Convert the list into a NumPy array of integers
    matrix = np.array(data_list, dtype=int)
    # Transpose the matrix
    transposed_matrix = matrix.T
    return transposed_matrix

design_values = transform_and_transpose(design_values)

for j in range(8):  #segmenti
    p = ax.bar(design_names, design_values[j], width=0.8, bottom=bottom, color = color[j])
    bottom += design_values[j]
    
for label in ax.get_xticklabels():
    label.set_ha("right")
    label.set_rotation(45)

plt.legend(label_list, loc='center left', bbox_to_anchor=(1, 0.5), title='Segments', title_fontsize='large')

# Set labels and title
ax.set_xlabel('Design')
ax.set_ylabel('Value')
ax.set_title('Stacked Bar Plot for Designs')

plt.savefig(f'{save_dir}/designs.png',bbox_inches="tight")
plt.savefig(f'{save_dir}/designs.pdf',bbox_inches="tight")




#reward
fig, ax = plt.subplots(figsize=(10, 6))
for i, (filename, _, values) in enumerate(optimized_design_list):
    color = colors(i)  # Select color for this design
    ax.bar(filename, values, width=0.8, color=color, label=filename)
plt.grid()
# Set labels and legend
ax.set_xlabel('Design')
ax.set_ylabel('Return')
ax.set_title('Segmented Heatmap Bar Plot')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
for label in ax.get_xticklabels():
    label.set_ha("right")
    label.set_rotation(45)
bottom, top = plt.ylim() 
plt.ylim(195,208)


plt.savefig(f'{save_dir}/reward.png',bbox_inches="tight")
plt.savefig(f'{save_dir}/reward.pdf',bbox_inches="tight")

podium_optimized_list = sorted_optimized_list[:3]
podium_random_list = sorted_random_list[:3]
podium_list = podium_optimized_list + podium_random_list
print(podium_list)

#comparison
fig, ax = plt.subplots(figsize=(10, 6))
for i, (filename, _, values) in enumerate(podium_list):
    color = colors(i)  # Select color for this design
    ax.bar(filename, values, width=0.8, color=color, label=filename)
plt.grid()
# Set labels and legend
ax.set_xlabel('Design')
ax.set_ylabel('Return')
ax.set_title('Segmented Heatmap Bar Plot')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
for label in ax.get_xticklabels():
    label.set_ha("right")
    label.set_rotation(45)
bottom, top = plt.ylim() 
plt.ylim(190,208)


plt.savefig(f'{save_dir}/comparison_reward.png',bbox_inches="tight")
plt.savefig(f'{save_dir}/comparison_reward.pdf',bbox_inches="tight")



