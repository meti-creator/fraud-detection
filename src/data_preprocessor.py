import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, filepath, target_col):
        self.filepath = filepath
        self.target_col = target_col
        self.df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset loaded successfully with shape: {self.df.shape}")
        return self.df

    def split_data(self, test_size=0.2, random_state=42):
        if self.df is None:
            self.load_data()
            
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Stratified split preserves the minority (fraud) class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test