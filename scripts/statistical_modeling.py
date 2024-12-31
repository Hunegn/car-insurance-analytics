import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import shap

class StatisticalModeling:
    def __init__(self, data_path):
        """
        Initialize the StatisticalModeling class with the dataset.
        Args:
            data_path (str): Path to the CSV file containing the dataset.
        """
        self.data = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        """
        Prepare the data for modeling: handle missing values, encode categorical features, and split the data.
        """
        print("Handling missing values...")
        imputer = SimpleImputer(strategy='mean')
        numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_features] = imputer.fit_transform(self.data[numeric_features])

        print("Encoding categorical data...")
        categorical_features = self.data.select_dtypes(include=['object', 'bool']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_features, drop_first=True)

        print("Splitting the data...")
        features = self.data.drop(columns=['TotalPremium', 'TotalClaims'])
        target = self.data['TotalPremium']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=0.3, random_state=42
        )

    def train_linear_regression(self):
        """
        Train a Linear Regression model and evaluate its performance.
        """
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.evaluate_model("Linear Regression", predictions)
        return model

    def train_random_forest(self):
        """
        Train a Random Forest model and evaluate its performance.
        """
        print("Training Random Forest model...")
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.evaluate_model("Random Forest", predictions)
        return model

    def train_xgboost(self):
        """
        Train an XGBoost model and evaluate its performance.
        """
        print("Training XGBoost model...")
        model = XGBRegressor(random_state=42)
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.evaluate_model("XGBoost", predictions)
        return model

    def evaluate_model(self, model_name, predictions):
        """
        Evaluate the performance of a model using RMSE and R-squared.
        Args:
            model_name (str): Name of the model being evaluated.
            predictions (array): Predictions made by the model.
        """
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f"{model_name} Performance:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        print("----------------------------------")

    def analyze_feature_importance(self, model):
        """
        Analyze feature importance using SHAP values.
        Args:
            model: The trained model to interpret.
        """
        print("Analyzing feature importance with SHAP...")
        explainer = shap.Explainer(model, self.X_test)
        shap_values = explainer(self.X_test)
        shap.summary_plot(shap_values, self.X_test)

    def main(self):
        """
        Execute the full statistical modeling pipeline.
        """
        self.prepare_data()
        
        lr_model = self.train_linear_regression()
        rf_model = self.train_random_forest()
        xgb_model = self.train_xgboost()

        self.analyze_feature_importance(rf_model)
        self.analyze_feature_importance(xgb_model)

if __name__ == "__main__":
    data_path = "../Data/clean/cleaned_insurance_data.csv"
    modeling = StatisticalModeling(data_path)
    modeling.main()
