# main.py
from src.data_preparation import load_data, preprocess_data
from src.model import train_model, evaluate_model, tune_model

def main():
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train a baseline model
    print("Training baseline model...")
    baseline_model = train_model(X_train, y_train)
    evaluate_model(baseline_model, X_test, y_test)
    
    # Tune the model
    print("\nTuning model...")
    tuned_model = tune_model(X_train, y_train)
    print("\nEvaluating tuned model...")
    evaluate_model(tuned_model, X_test, y_test)

if __name__ == "__main__":
    main()
