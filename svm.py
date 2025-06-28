# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def run_iris_example():
    """Run SVM classification on Iris dataset"""
    try:
        print("=" * 50)
        print("IRIS DATASET CLASSIFICATION")
        print("=" * 50)
        data = load_iris()
        X, y = data.data, data.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=data.target_names))
        return True
    except Exception as e:
        print(f"Error in Iris classification: {str(e)}")
        return False

def run_tomato_analysis(csv_file='tomatoes.csv'):
    """Run SVM classification on tomato dataset"""
    try:
        print("\n" + "=" * 50)
        print("TOMATO DATASET CLASSIFICATION")
        print("=" * 50)
        try:
            df = pd.read_csv(csv_file)
            print(f"Dataset loaded successfully with shape: {df.shape}")
        except FileNotFoundError:
            print(f"Warning: '{csv_file}' not found. Creating sample data for demonstration.")
            df = create_sample_tomato_data()
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            print("Creating sample data for demonstration.")
            df = create_sample_tomato_data()
        print("\nFirst 5 rows of dataset:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing values:")
        missing_values = df.isnull().sum()
        print(missing_values)
        if missing_values.sum() > 0:
            print("Handling missing values...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        required_cols = ['Weight', 'Color', 'Firmness', 'Ripeness']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return False
        df_processed = df.copy()
        label_encoders = {}
        if df_processed['Color'].dtype == 'object':
            label_encoders['Color'] = LabelEncoder()
            df_processed['Color'] = label_encoders['Color'].fit_transform(df_processed['Color'].astype(str))
        if df_processed['Ripeness'].dtype == 'object':
            label_encoders['Ripeness'] = LabelEncoder()
            df_processed['Ripeness_Label'] = label_encoders['Ripeness'].fit_transform(df_processed['Ripeness'].astype(str))
            target_names = label_encoders['Ripeness'].classes_
        else:
            df_processed['Ripeness_Label'] = df_processed['Ripeness']
            target_names = [f"Class_{i}" for i in sorted(df_processed['Ripeness_Label'].unique())]
        feature_cols = ['Weight', 'Color', 'Firmness']
        X = df_processed[feature_cols].select_dtypes(include=[np.number])
        y = df_processed['Ripeness_Label']
        if X.empty or y.empty:
            print("Error: No valid numerical features found for modeling.")
            return False
        if len(X) < 10:
            print("Warning: Very small dataset. Results may not be reliable.")
        X = X.fillna(X.median())
        y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            print("Error: Need at least 2 classes for classification.")
            return False
        test_size = min(0.3, max(0.1, 1 - (5 / len(X))))
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout();
            plt.show();
        except Exception as e:
            print(f"Could not plot confusion matrix: {str(e)}")
        return True
    except Exception as e:
        print(f"Error in tomato analysis: {str(e)}")
        return False

def create_sample_tomato_data():
    """Create sample tomato data for demonstration"""
    np.random.seed(42)
    n_samples = 150
    data = {
        'Weight': np.random.normal(100, 20, n_samples),
        'Color': np.random.choice(['Red', 'Green', 'Yellow'], n_samples),
        'Firmness': np.random.normal(5, 1.5, n_samples),
        'Ripeness': np.random.choice(['Unripe', 'Ripe', 'Overripe'], n_samples)
    }
    data['Weight'] = np.abs(data['Weight'])
    data['Firmness'] = np.clip(data['Firmness'], 1, 10)
    df = pd.DataFrame(data)
    print("Sample tomato dataset created for demonstration.")
    return df

def main():
    """Main function to run both analyses"""
    print("Running SVM Classification Analysis")
    print("=" * 60)
    iris_success = run_iris_example()
    tomato_success = run_tomato_analysis()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Iris analysis: {'✓ Success' if iris_success else '✗ Failed'}")
    print(f"Tomato analysis: {'✓ Success' if tomato_success else '✗ Failed'}")

if __name__ == "__main__":
    main()
        