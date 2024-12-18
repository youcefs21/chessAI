import numpy as np
import keras
from training import setup_logging, ChessDataFrame, preprocess_data, Sizes, ChessRNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Setup logging
logger = setup_logging()

def load_model(model_path="best_model.keras", sequence_length=30):
    """
    Load the trained RNN model weights into a new ChessRNN instance
    """
    logger.info(f"Loading model from {model_path}")
    try:
        # Create a new ChessRNN instance
        rnn = ChessRNN(sequence_length=sequence_length)
        
        # Load some sample data to build the model architecture
        chess_df = ChessDataFrame(size=Sizes.smol)
        games, _ = preprocess_data(chess_df.df_train.head(1))
        X, _ = rnn.prepare_sequences(games, [0])
        
        # Build the model with the correct input shape
        input_shape = (sequence_length, X.shape[2])
        rnn.build_model(input_shape)
        
        # Load the trained weights
        rnn.model.load_weights(model_path)
        logger.info("Model loaded successfully")
        return rnn
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def evaluate_model(rnn, data, title="Model Evaluation", plot=True):
    """
    Comprehensive evaluation of the model including metrics and visualizations
    """
    logger.info("Starting model evaluation...")

    # Preprocess test data and get valid indices
    games, true_labels = preprocess_data(data)
    X, y = rnn.prepare_sequences(games, true_labels)

    # Get predictions for valid sequences only
    predictions_prob = rnn.model.predict(X)
    predictions = (predictions_prob > 0.5).astype(int).flatten()
    y = y.flatten()

    # Now true_labels and predictions will have matching lengths
    conf_matrix = confusion_matrix(y, predictions)
    class_report = classification_report(y, predictions)

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y, predictions_prob.flatten())
    roc_auc = auc(fpr, tpr)

    if plot:
        # Create visualization
        plt.figure(figsize=(15, 5))

        # Plot 1: Confusion Matrix
        plt.subplot(131)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{title}\nConfusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Plot 2: ROC Curve
        plt.subplot(132)
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")

        # Plot 3: Prediction Distribution
        plt.subplot(133)
        sns.histplot(predictions_prob, bins=50)
        plt.title("Prediction Distribution")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()

    # Print classification report and metrics
    logger.info("\nClassification Report:\n" + class_report)
    logger.info(f"ROC AUC Score: {roc_auc:.3f}")

    # Calculate additional metrics
    accuracy = (predictions == y).mean()
    logger.info(f"Accuracy: {accuracy:.3f}")

    # Log number of samples used
    logger.info(f"Evaluated on {len(y)} samples (filtered from {len(true_labels)} total games)")

    return {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "predictions": predictions,
        "probabilities": predictions_prob.flatten(),
        "n_samples": len(y),
        "n_total_games": len(true_labels),
    }


if __name__ == "__main__":
    # Load model
    rnn = load_model()
    
    # Load some test data
    chess_df = ChessDataFrame(size=Sizes.mid)
    logger.info("Successfully loaded test data")
    
    # Make predictions
    metrics = evaluate_model(rnn, chess_df.df_test, "Chess RNN Evaluation")
    logger.info(metrics)
