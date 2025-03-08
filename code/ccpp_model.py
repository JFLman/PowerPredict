import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import pickle

def load_data(file_path):
    """Load and validate CCPP dataset."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded:", data.columns.tolist())
        return data
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        exit(1)

def prepare_data(data):
    """Prepare features and target."""
    X = data[['T', 'AP', 'RH', 'V']]  # Adjust if column names differ
    y = data['PE']
    return X, y

def train_and_evaluate(X, y):
    """Train and evaluate models, return best model."""
    # Data split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_val_pred = lr.predict(X_val)
    lr_rmse = np.sqrt(mean_squared_error(y_val, lr_val_pred))
    print(f"Linear Regression Validation RMSE: {lr_rmse:.2f} MW")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
    print(f"Random Forest Validation RMSE: {rf_rmse:.2f} MW")

    # Select best model
    best_model = rf if rf_rmse < lr_rmse else lr
    print(f"Best model: {'Random Forest' if rf_rmse < lr_rmse else 'Linear Regression'}")

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Model saved to model.pkl")

    # Test evaluation
    test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    print(f"Test RMSE: {test_rmse:.2f} MW")

    # Feature importance (if Random Forest)
    if isinstance(best_model, RandomForestRegressor):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        print("Feature Importances:\n", importances.sort_values(ascending=False))

    return best_model, X_test, y_test, test_pred

def plot_results(y_test, test_pred):
    """Generate and save prediction plot."""
    plt.scatter(y_test, test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual PE (MW)')
    plt.ylabel('Predicted PE (MW)')
    plt.title('PowerPredict: Predicted vs Actual Output')
    plt.savefig('assets/model_demo.png', dpi=300)
    print("Plot saved to assets/model_demo.png")

def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description="PowerPredict: AI Energy Output Prediction")
    parser.add_argument('--data', default='data/CCPP_data.csv', help='Path to dataset')
    args = parser.parse_args()

    data = load_data(args.data)
    X, y = prepare_data(data)
    best_model, X_test, y_test, test_pred = train_and_evaluate(X, y)
    plot_results(y_test, test_pred)

if __name__ == "__main__":
    main()