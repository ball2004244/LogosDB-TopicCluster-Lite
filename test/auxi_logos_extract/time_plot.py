import matplotlib.pyplot as plt
import numpy as np

# Define subjects and scores
subjects = [
    'abstract_algebra',
    'college_physics',
    'electrical_engineering',
    'high_school_biology',
    'machine_learning',
    'high_school_chemistry',
    'high_school_geography',
    'sociology',
    'high_school_macroeconomics',
    'professional_psychology',
    'human_sexuality',
    'public_relations',
    'high_school_world_history',
    'logical_fallacies',
    'world_religions',
    'philosophy',
    'professional_law',
    'moral_scenarios'
]
scores = [2593.7906, 2668.8363, 2647.5657, 7229.2455, 2519.2535, 4481.3546,
          3824.0641, 4019.0867, 7888.1507, 12751.7298, 2367.3514, 2191.5974,
          5460.5902, 3408.8274, 3259.0034, 6369.3096, 36490.7987, 4030.4995*5]

# Create indices for the x-axis
indices = np.arange(len(subjects))

# Plot the bar chart
plt.figure(figsize=(16,10))
plt.bar(indices[:6], scores[:6], color='blue', label='Natural Science')
plt.bar(indices[6:12], scores[6:12], color='green', label='Social Science')
plt.bar(indices[12:], scores[12:], color='red', label='Humanities')

# Add labels and title
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.title('Runtime for Subjects')
plt.xticks(indices, subjects, rotation=30, ha='right')

# Add legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig('test/auxi_logos_extract/figures/time_plot.png')
print('Saved plot to test/auxi_logos_extract/figures/time_plot.png')