#plot designs

import numpy as np
import sys
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

design_folder_path = "data_exp_sac_pso_batch"

design_path = sys.argv[1]

design_folder_path = f"{design_folder_path}/{design_path}"
save_dir=(f"{design_folder_path}/evaluation")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def read_second_row_as_int(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        if len(rows) >= 2:
            return [int(cell) for cell in rows[1]]

def read_csv_files_in_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.startswith("data_design_") and filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            second_row = read_second_row_as_int(file_path)
            if second_row:
                # Extract number from filename
                file_number = int(filename.split("_")[2].split(".")[0])
                data.append((file_number, filename, second_row))
    # Sort data based on the number in filename
    data.sort(key=lambda x: x[0])
    return [(filename, second_row) for _, filename, second_row in data]

designs_list = read_csv_files_in_folder(design_folder_path)

#-------------------discard random and first five designs------------------------------
designs_list_optimized = []
for idx in range(len(designs_list)):
    if idx > 4 and idx%2 == 0:
        designs_list_optimized.append(designs_list[idx])
designs_list = designs_list_optimized

## Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each design as separate bars with different colors
num_designs = len(designs_list)
num_segments = len(designs_list[0][1])
bar_width = 0.8 / num_designs

# Create a list of colors, one for each design
colors = plt.cm.get_cmap('turbo', num_designs)

for i, (filename, values) in enumerate(designs_list):
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
design_names = [design[0] for design in designs_list]
design_values = [design[1] for design in designs_list]

bottom = np.zeros(len(designs_list))
bottom_array = []
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