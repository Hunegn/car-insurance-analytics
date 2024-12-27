import pandas as pd

def load_data(file_path):
    """
    Load the raw data into a pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)
def clean_data(data):
    """
    Clean the dataset by handling missing values and anomalies.
    Args:
        data (pd.DataFrame): The raw dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    data = data.drop_duplicates()
    
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    
    return data
def save_cleaned_data(data, output_path):
    """
    Save cleaned data to a CSV file.
    Args:
        data (pd.DataFrame): Cleaned dataset.
        output_path (str): Path to save the file.
    """
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
def main():
   
    input_path = "../Data/raw/MachineLearningRating_v3.csv"  
    data = load_data(input_path)

    cleaned_data = clean_data(data)

    output_path = "../Data/clean/cleaned_insurance_data.csv"
    save_cleaned_data(cleaned_data, output_path)

    return cleaned_data

if __name__ == "__main__":
    main()