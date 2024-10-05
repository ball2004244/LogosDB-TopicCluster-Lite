import matplotlib.pyplot as plt
import pandas as pd

def double_column_bar_chart(data, save_path, title='Double Column Bar Chart', xlabel='X-axis', ylabel='Y-axis'):
    """
    Creates a double column bar chart based on the provided dataset.

    Parameters:
    data (pandas.DataFrame): The dataset containing the columns to plot.
    column1 (str): The name of the first column to plot.
    column2 (str): The name of the second column to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """
    plt.figure(figsize=(16, 10))
    
    x = range(len(data))
    plt.ylim(2000, 7000)
    plt.bar(x, data['smart'], width=0.4, label='smart', color='b', align='center')
    plt.bar([p + 0.4 for p in x], data['improved'], width=0.4, label='improved', color='r', align='center')
    
    plt.title('Time plot for college subjects')
    plt.xlabel('Subject')
    plt.ylabel('Time (s)')
    plt.xticks([p + 0.2 for p in x], data['subject'], rotation=45)
    plt.legend()
    
    plt.savefig(f'{save_path}/time_plot.png')
    print(f'Saved figure to {save_path}/time_plot.png')

save_path = 'test/query_testing/figures'
data = pd.DataFrame({
    'subject': [
        'college_computer_science',
        'college_biology',
        'college_chemistry',
        'college_physics',
        'college_mathematics',
        'college_medicine'
    ],
    'smart': [2466.6363, 2417.2874, 2321.9066, 2448.9316, 3380.5955, 3308.3574],
    'improved': [2716.7526, 2768.5868, 2559.8962, 4705.7605, 6561.1265, 3464.2936]
})
double_column_bar_chart(data, save_path)