
import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import IsolationForest
from scipy.stats.mstats import winsorize
from sklearn.compose import ColumnTransformer

from torch.utils.data import Dataset

import pandas as pd
from scipy.stats import boxcox


class Dataset(Dataset):
    def __init__(self, data_path, target_col):
        self.data_path = data_path
        self.target_col = target_col

        self.preprocessed_data_df = None
        self.data_df = None
        self.data_df_with_outliers = None
        
        self.X = None
        self.target = None
        
        self.X_train= None
        self.X_test = None
        self.target_train = None
        self.target_test = None

        self.num_features = None

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        sample = (self.data_df[idx], self.target[idx])
        return sample
    
    def load_data(self):
        self.data_df = pd.read_csv(self.data_path)
        return None
    
    def show_data(self):
        return self.data_df
    
    def separate_features_and_target(self, data_df=None):
        if data_df is None:
            data_df = self.data_df

        self.X = data_df.drop(self.target_col, axis=1)
        self.target = data_df[self.target_col]
        return None

    def preprocess_data(self, drop_duplicates=True, handle_outliers='winsorize', handle_missing_values=True, encode_categorical=True, test_size = 0.33,\
                         random_state=45, pca = True, pca_k = 25, scaler = 'standard_scaler', feature_selection = True, features_score_plot=False):

        self.preprocessed_data_df = self.data_df.copy()
        
        # Drop duplicates
        if drop_duplicates:
            self.preprocessed_data_df = self.preprocessed_data_df.drop_duplicates()


        upper=self.preprocessed_data_df['Credit_Limit'].quantile(0.25)
        lower=self.preprocessed_data_df['Credit_Limit'].quantile(0.75)

        index=self.preprocessed_data_df[(self.preprocessed_data_df['Credit_Limit']>=lower)|(self.preprocessed_data_df['Credit_Limit']<=upper)].index

        self.preprocessed_data_df.drop(index,inplace=True)



        self.separate_features_and_target(self.preprocessed_data_df)

        if handle_missing_values:
            self.X = self.fix_missing_values_and_cat_encode(self.X, encode_categorical=encode_categorical)
            self.preprocessed_data_df = self.fix_missing_values_and_cat_encode(self.preprocessed_data_df, encode_categorical=encode_categorical)
            self.encoded = self.fix_missing_values_and_cat_encode(self.preprocessed_data_df, encode_categorical=encode_categorical)
        self.data_df_with_outliers = self.detect_outliers_rf(self.preprocessed_data_df)


        if handle_outliers == 'winsorize':
            self.X = self.handle_outlier_winsorize(self.X)
            self.target = self.handle_outlier_winsorize(pd.DataFrame(self.target), lower_percentile=.2, upper_percentile=.2)
            self.preprocessed_data_df = self.handle_outlier_winsorize(self.preprocessed_data_df)
        elif handle_outliers == 'log_transform':
            self.X = self.handle_outlier_log_transform(self.X)
            # self.target = self.handle_outlier_log_transform(pd.DataFrame(self.target))
            # self.preprocessed_data_df = self.handle_outlier_log_transform(self.preprocessed_data_df)

        else:
            pass
        
        self.X_train, self.X_test, self.target_train, self.target_test = train_test_split(self.X, self.target, test_size=test_size, random_state=random_state)
        
        if scaler == 'standard_scaler':
            scaler = StandardScaler()
        elif scaler == 'min_max_scaler':
            scaler = MinMaxScaler()

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        if pca:
            pca = PCA(pca_k)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
        
        if feature_selection:
            num_features_to_keep = self.calculate_features_score(show_plot=features_score_plot)
            self.X_train, self.X_test, _ = self.select_features(k=num_features_to_keep)

        self.num_features = self.X_train.shape[1]

        return None
    
    def fix_missing_values_and_cat_encode(self, data_df, inplace=False, encode_categorical = True):

        if not inplace:
            data_tmp = data_df.copy()
        else:  
            data_tmp = data_df

        # Identify numerical and categorical columns
        num_cols = data_tmp.select_dtypes(exclude='object').columns
        cat_cols = data_tmp.select_dtypes(include='object').columns

        # Create transformers for numerical and categorical data
        num_transformer = IterativeImputer()

        if encode_categorical: 
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        else:
            cat_transformer = SimpleImputer(strategy='most_frequent')

        # Combine transformers into a ColumnTransformer
        fix_missings_and_cat_encode_preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )

        data_tmp = fix_missings_and_cat_encode_preprocessor.fit_transform(data_tmp)

        data_tmp = pd.DataFrame(data_tmp)

        return data_tmp

    def detect_outliers_rf(self, data_df, inplace=False):
        if not inplace:
            data_tmp = data_df.copy()
        else:
            data_tmp = data_df

        clf = IsolationForest(contamination=0.1)
        clf.fit(data_tmp)
        outlier_mask = clf.predict(data_tmp) == -1
        data_tmp['is_outlier'] = outlier_mask

        return data_tmp
    
    def handle_outlier_winsorize(self, data_df, lower_percentile=0.05, upper_percentile=0.05, inplace=False):
        """
        Apply Winsorization to all columns in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame to be Winsorized.
        lower_percentile (float): The lower percentile limit for Winsorization.
        upper_percentile (float): The upper percentile limit for Winsorization.
        
        Returns:
        pd.DataFrame: The Winsorized DataFrame.
        """
        if not inplace:
            data_df_tmp = data_df.copy()
        else: 
            data_df_tmp = data_df

        # Winsorize all columns in the DataFrame
        df_winsorized = data_df_tmp.apply(lambda x: winsorize(x, limits=(lower_percentile, upper_percentile)))
        
        # Convert the winsorized array back to DataFrame
        df_winsorized = pd.DataFrame(df_winsorized, columns=data_df_tmp.columns)
        
        data_df_tmp = df_winsorized

        return data_df_tmp
    
    def handle_outlier_log_transform(self, data_df, inplace=False):
        """
        Apply Log Transformation to all numeric columns in the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame to be log-transformed.
        
        Returns:
        pd.DataFrame: The log-transformed DataFrame.
        """

        if not inplace:
            data_df_tmp = data_df.copy()
        else: 
            data_df_tmp = data_df

        df_log_transformed = data_df.apply(lambda x: np.log1p(x) if np.issubdtype(x.dtype, np.number) else x)
        
        return df_log_transformed

    def select_features(self, k = 25):
        fs = SelectKBest(score_func=mutual_info_regression, k=k)
        fs.fit(self.X_train, self.target_train)
        X_train_fs = fs.transform(self.X_train)

        X_test_fs = fs.transform(self.X_test)

        return X_train_fs, X_test_fs, fs
    
    def calculate_features_score(self, show_plot = False):
        _, num_features = self.X_train.shape
        _, _, fs = self.select_features(k=num_features)

        ignr_cnt = 0
        
        for i in range(len(fs.scores_)):
            if (fs.scores_[i] < 0.0015):
                ignr_cnt +=1

                if show_plot:
                    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
                    plt.title('Feature Score Over Feature Indexes')
                    plt.xlabel('Feature Index')
                    plt.ylabel('Feature Score')
                    
        return num_features - ignr_cnt

    def prepare_for_model(self):
        self.X_train = np.array(self.X_train) 
        self.X_test = np.array(self.X_test) 

        self.target_train = np.array(self.target_train) 
        self.target_test = np.array(self.target_test) 
        
        self.X_train = torch.from_numpy(self.X_train.astype(np.float32))
        self.X_test = torch.from_numpy(self.X_test.astype(np.float32))
        self.target_train = torch.from_numpy(self.target_train.astype(np.float32))
        self.target_test = torch.from_numpy(self.target_test.astype(np.float32))



        self.target_train = self.target_train.view(self.target_train.shape[0], 1)
        self.target_test = self.target_test.view(self.target_test.shape[0], 1)

    def to_device(self, device):
        self.X_train = self.X_train.to(device)
        self.X_test = self.X_test.to(device)
        self.target_train = self.target_train.to(device)
        self.target_test = self.target_test.to(device)


