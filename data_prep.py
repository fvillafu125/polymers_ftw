### 1. Import necessary libraries ###

import pandas as pd
import numpy as np
from rdkit import Chem
import random
import tqdm
from tqdm import tqdm
from typing import Optional, List, Union

### 2. Read polymer data from Kaggle datasets ###

train = pd.read_csv('data/neurips-open-polymer-prediction-2025/train.csv')
# test = pd.read_csv('data/neurips-open-polymer-prediction-2025/test.csv')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# 2.1 Supplement train and test data with Kaggle supplemental datasets

for i in range(1, 5):
    supplement_path = f'data/neurips-open-polymer-prediction-2025/train_supplement/dataset{i}.csv'
    supplement_ds = pd.read_csv(supplement_path)

    if 'TC_mean' in supplement_ds.columns:
        supplement_ds = supplement_ds.rename(columns = {'TC_mean': 'Tc'})

    train = pd.concat([train, supplement_ds], axis=0)

train = train.sample(frac=1).reset_index(drop=True)
# print(f"Training set columns pre-augmentation are: {train.columns.tolist()}")

# 2.2 Check the to see if Tc has been added correctly from the supplemental datasets
train
train['Tc']


### 3. Augment the dataset using Chem from rdkit ###

def augment_smiles_dataset(df: pd.DataFrame,
                               smiles_column: str = 'SMILES',
                               augmentation_strategies: List[str] = ['enumeration', 'kekulize', 'stereo_enum'],
                               n_augmentations: int = 10,
                               preserve_original: bool = True,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Advanced SMILES augmentation with multiple strategies.
    
    Parameters:
    -----------
    augmentation_strategies : List[str]
        List of augmentation strategies: 'enumeration', 'kekulize', 'stereo_enum'
    """
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def apply_augmentation_strategy(smiles: str, strategy: str) -> List[str]:
        """Apply specific augmentation strategy"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [smiles]
            
            augmented = []
            
            if strategy == 'enumeration':
                # Standard SMILES enumeration
                for _ in range(n_augmentations):
                    enum_smiles = Chem.MolToSmiles(mol, 
                                                 canonical=False, 
                                                 doRandom=True,
                                                 isomericSmiles=True)
                    augmented.append(enum_smiles)
            
            elif strategy == 'kekulize':
                # Kekulization variants
                try:
                    Chem.Kekulize(mol)
                    kek_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
                    augmented.append(kek_smiles)
                except:
                    pass
            
            elif strategy == 'stereo_enum':
                # Stereochemistry enumeration
                for _ in range(n_augmentations // 2):
                    # Remove stereochemistry
                    Chem.RemoveStereochemistry(mol)
                    no_stereo = Chem.MolToSmiles(mol)
                    augmented.append(no_stereo)
            
            return list(set(augmented))  # Remove duplicates
            
        except Exception as e:
            print(f"Error in {strategy} for {smiles}: {e}")
            return [smiles]
    
    augmented_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        original_smiles = row[smiles_column]
        
        # Add original if requested
        if preserve_original:
            original_row = row.to_dict()
            original_row['augmentation_strategy'] = 'original'
            original_row['is_augmented'] = False
            augmented_rows.append(original_row)
        
        # Apply each augmentation strategy
        for strategy in augmentation_strategies:
            strategy_smiles = apply_augmentation_strategy(original_smiles, strategy)
            
            for aug_smiles in strategy_smiles:
                if aug_smiles != original_smiles:  # Avoid duplicating original
                    new_row = row.to_dict().copy()
                    new_row[smiles_column] = aug_smiles
                    new_row['augmentation_strategy'] = strategy
                    new_row['is_augmented'] = True
                    augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"Advanced augmentation completed:")
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")
    print(f"Augmentation factor: {len(augmented_df) / len(df):.2f}x")
    
    return augmented_df

def create_splits(df):
    length = len(df)
    train_length = int(0.9 * length)
    train = df.loc[:train_length]
    test = df.loc[train_length:]
    return train, test

train, test = create_splits(train)

train = augment_smiles_dataset(train)
test = augment_smiles_dataset(test)

# # Only keep rows where columns 2:6 are all filled (no NaN, None, or empty/space strings)
# target_cols = train.columns[2:7]  # columns 2,3,4,5,6

# # Replace 'None' and ' ' with np.nan for consistency
# train[target_cols] = train[target_cols].replace(['None', ' ', ''], np.nan)
# test[target_cols] = test[target_cols].replace(['None', ' ', ''], np.nan)

# # Drop rows with any missing values in target columns
# train = train.dropna(subset=target_cols)
# test = test.dropna(subset=target_cols)

print("Training set columns are: ",train.columns)
print("Training set shape is: ",train.shape)
print("Test set columns are: ",test.columns)
print("Test set shape is: ",test.shape)

# Drop columns by index: 0 (first), 7 and 8 (last two)
cols_to_drop = [train.columns[0], train.columns[7], train.columns[8]]
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

# After dropping columns and before saving, create individual datasets for each target property

smiles_col = train.columns[0]  # SMILES column (after dropping columns)
target_cols = train.columns[1:6]  # Target property columns

for target in target_cols:
    train_target_df = train[[smiles_col, target]].dropna()
    test_target_df = test[[smiles_col, target]].dropna()
    train_target_df.to_csv(f'data/neurips_augmented_train_{target}.csv', index=False)
    test_target_df.to_csv(f'data/neurips_augmented_test_{target}.csv', index=False)

# train.to_csv('data/neurips_augmented_train.csv', index=False)
# test.to_csv('data/neurips_augmented_test.csv', index=False)