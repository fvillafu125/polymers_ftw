import pandas as pd

data_file1 = "data/neurips_augmented_train_Tg.csv"
data_file2 = "data/neurips_augmented_test_Tg.csv"
save_file1 = "data/neurips_augmented_train_Tg_znorm.csv"
save_file2 = "data/neurips_augmented_test_Tg_znorm.csv"

# Replace 'your_data.csv' with the path to your dataset
df1 = pd.read_csv(data_file1)
df2 = pd.read_csv(data_file2)

# Convert to Kelvin and find mean, standard deviation, and z score

df1['Tg'] = df1['Tg'] + 273.15
df2['Tg'] = df2['Tg'] + 273.15
alltg = pd.concat([df1, df2])
mean_tg = alltg['Tg'].mean()
std_tg = alltg['Tg'].std()
# df1_mean = df1['Tg'].mean()
# df1_std = df1['Tg'].std()
# df2_mean = df2['Tg'].mean()
# df2_std = df2['Tg'].std()

print(f"Mean Tg across both training and test sets is {mean_tg}.")
print(f"Tg standard deviation across training and test sets is {std_tg}.")
# print(f"Mean train Tg is {df1_mean}.")
# print(f"Train Tg standard deviation is {df1_std}.")
# print(f"Mean test Tg is {df2_mean}.")
# print(f"Test Tg standard deviation is {df2_std}.")

# Calculate z norm for both data sets

df1['Tg'] = (df1['Tg'] - mean_tg) / std_tg
df2['Tg'] = (df2['Tg'] - mean_tg) / std_tg
# df1['Tg'] = (df1['Tg'] - df1_mean) / df1_std
# df2['Tg'] = (df2['Tg'] - df2_mean) / df2_std

# Save the result to a new CSV file
df1.to_csv(save_file1, index=False)
df2.to_csv(save_file2, index=False)

print(f"Normalization complete. Output saved to {save_file1} and {save_file2}.")