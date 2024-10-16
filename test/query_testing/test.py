import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create a pandas DataFrame with the data
data = {
    'Subjects': ['Math', 'Science', 'English', 'History', 'Art', 'PE'],
    'Category1_Means': [0.6, 0.7, 0.5, 0.6, 0.7, 0.8],
    'Category2_Means': [0.5, 0.6, 0.4, 0.5, 0.6, 0.7],
    'Category1_Errors': [0.05, 0.04, 0.06, 0.05, 0.04, 0.03],
    'Category2_Errors': [0.04, 0.05, 0.05, 0.04, 0.03, 0.02]
}

df = pd.DataFrame(data)

# Extract the data for plotting
subjects = df['Subjects']
category1_means = df['Category1_Means']
category2_means = df['Category2_Means']
category1_errors = df['Category1_Errors']
category2_errors = df['Category2_Errors']

# X locations for the groups
x = np.arange(len(subjects))

# Create a figure and axis
fig, ax = plt.subplots()

# Plot error bars for category 1
ax.errorbar(x - 0.1, category1_means, yerr=category1_errors, fmt='o', label='Category 1', capsize=5)

# Plot error bars for category 2
ax.errorbar(x + 0.1, category2_means, yerr=category2_errors, fmt='o', label='Category 2', capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Subjects')
ax.set_ylabel('Scores')
ax.set_title('Scores by subject and category')
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.legend()

# rotate x
ax.set_xticklabels(subjects, rotation=45)

print(df)

# Save the plot
save_path = 'test'
plt.savefig(os.path.join(save_path, 'aggr_accuracy.png'))
