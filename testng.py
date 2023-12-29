import csv

import pandas as pd

label_mapping = {
    'Appreciative': 0, 'Cautionary': 1, 'Diplomatic': 2, 'Direct': 3, 'Informative': 4,
    'Inspirational': 5, 'Thoughtful': 6, 'Witty': 7, 'Absurd': 8, 'Accusatory': 9,
    'Acerbic': 10, 'Admiring': 11, 'Aggressive': 12, 'Aggrieved': 13, 'Altruistic': 14,
    'Ambivalent': 15, 'Amused': 16, 'Angry': 17, 'Animated': 18, 'Apathetic': 19,
    'Apologetic': 20, 'Ardent': 21, 'Arrogant': 22, 'Assertive': 23, 'Belligerent': 24,
    'Benevolent': 25, 'Bitter': 26, 'Callous': 27, 'Candid': 28, 'Caustic': 29
}

with open('OpenChat/tone_v1.txt', 'r') as file:
    data = file.readlines()

# data = [(line.split('||')[0].strip(), line.split('||')[1].strip()) for line in lines]



# Splitting data and mapping labels
split_data = [item.split(' || ') for item in data]
sentences = [item[0] for item in split_data]
labels = [item[1].strip() for item in split_data]

numerical_labels = [label_mapping.get(label, -1) for label in labels]
# print(numerical_labels)

# Creating a DataFrame using pandas
df = pd.DataFrame({'Sentence': sentences, 'Label': labels, 'Numerical Label': numerical_labels})

# Writing to a CSV file
csv_filename = 'OpenChat/labeled_data.csv'
df.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' created successfully.")