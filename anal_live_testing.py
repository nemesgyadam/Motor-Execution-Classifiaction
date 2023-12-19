import pandas as pd
from pprint import pprint

# Load the CSV data into a DataFrame
data = pd.read_csv('../0717b399_live_testing_results.txt')

# Calculate accuracy for each relevant column
accuracy_10sec = data['10sec'].mean()
accuracy_30sec = data['30sec'].mean()
accuracy_dominan1 = data['dominan1'].mean()

# Calculate accuracies per direction (n, l, r) for each relevant column
accuracy_per_dir_10sec = data.groupby('dir')['10sec'].mean()
accuracy_per_dir_30sec = data.groupby('dir')['30sec'].mean()
accuracy_per_dir_dominan1 = data.groupby('dir')['dominan1'].mean()

# Create a summary dictionary
print(f'n = {data.shape[0]}')
summary_statistics = {
    'Overall': {
        '10sec': accuracy_10sec,
        '30sec': accuracy_30sec,
        'dominan1': accuracy_dominan1
    },
    'Per Direction': {
        '10sec': accuracy_per_dir_10sec.to_dict(),
        '30sec': accuracy_per_dir_30sec.to_dict(),
        'dominan1': accuracy_per_dir_dominan1.to_dict()
    }
}

pprint(summary_statistics)

