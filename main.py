# main.py
from src.data_preparation import load_data, preprocess_data
from src.model import train_model, evaluate_model, tune_model
import pickle

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

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/tuned_model.pkl', 'wb') as f:
        pickle.dump(tuned_model, f)

if __name__ == "__main__":
    main()
