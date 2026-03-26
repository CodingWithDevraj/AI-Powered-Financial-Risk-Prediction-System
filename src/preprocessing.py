import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    
    df['Loan_Income_Ratio'] = df['Loan_Amount'] / df['Income']
    df['EMI'] = df['Loan_Amount'] / df['Loan_Term']
    
    df = pd.get_dummies(df, drop_first=True)
    
    return df