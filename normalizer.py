import pandas as pd

data_file1 = "data/neurips_augmented_train_Tg.csv"
data_file2 = "data/neurips_augmented_test_Tg.csv"
save_file1 = "data/neurips_augmented_train_Tg_norm.csv"
save_file2 = "data/neurips_augmented_test_Tg_norm.csv"

# Replace 'your_data.csv' with the path to your dataset
df1 = pd.read_csv(data_file1)
df2 = pd.read_csv(data_file2)

# Convert to Kelvin and find the max Tg value

df1['Tg'] = df1['Tg'] + 273.15
df2['Tg'] = df2['Tg'] + 273.15
alltg = pd.concat([df1, df2])
max_tg = alltg['Tg'].max()
print(f"Max Tg across both training and test sets is {max_tg}.")

# Normalize Tg values by dividing by the maximum
df1['Tg'] = df1['Tg'] / max_tg
df2['Tg'] = df2['Tg'] / max_tg

# Save the result to a new CSV file
df1.to_csv(save_file1, index=False)
df2.to_csv(save_file2, index=False)

print(f"Normalization complete. Output saved to {save_file1} and {save_file2}.")