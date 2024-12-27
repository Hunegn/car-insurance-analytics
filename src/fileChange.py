import pandas as pd

input_file = '../Data/MachineLearningRating_v3.txt'
output_file = '../Data/MachineLearningRating_v3.csv'

df = pd.read_csv(input_file, delimiter='\t')  # Change '\t' to ',' or ' ' based on your file
df.to_csv(output_file, index=False)

print(f"File converted and saved as {output_file}")
