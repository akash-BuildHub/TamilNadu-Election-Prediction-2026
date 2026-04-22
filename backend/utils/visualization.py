"""
Visualization Utilities for Election Prediction
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os


def setup_style():
    """Set up matplotlib style for consistent visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def get_party_colors() -> Dict[str, str]:
    """Get standard colors for Kerala political parties"""
    return {
        'LDF': '#FF0000',      # Red for Left Front
        'UDF': '#0066CC',      # Blue for Congress alliance
        'NDA': '#FF9933',      # Saffron for BJP alliance
        'OTHERS': '#808080'    # Gray for others
    }


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (parties)
        save_path: Optional path to save
        normalize: If True, normalize by true labels
    
    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import confusion_matrix
    
    setup_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = get_party_colors()
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Party')
    ax.set_ylabel('True Party')
    ax.set_title('Kerala Election Prediction - Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_prediction_distribution(
    probabilities: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of prediction probabilities.
    
    Args:
        probabilities: Prediction probabilities (N, num_classes)
        class_names: Names of classes
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    setup_style()
    colors = get_party_colors()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, party in enumerate(class_names):
        color = colors.get(party, '#808080')
        ax.hist(
            probabilities[:, i],
            bins=50,
            alpha=0.6,
            label=party,
            color=color
        )
    
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prediction Probabilities by Party')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature/modality importance.
    
    Args:
        importance: Dictionary of feature name to importance score
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    features = list(importance.keys())
    values = list(importance.values())
    
    colors = ['#4CAF50', '#2196F3', '#FF9800'][:len(features)]
    
    bars = ax.barh(features, values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2%}',
            va='center'
        )
    
    ax.set_xlabel('Importance')
    ax.set_title('Feature Modality Importance')
    ax.set_xlim(0, max(values) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_district_predictions(
    predictions: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot predictions aggregated by district.
    
    Args:
        predictions: DataFrame with 'district', 'prediction', 'probability' columns
        save_path: Optional path to save
    
    Returns:
        matplotlib Figure
    """
    setup_style()
    colors = get_party_colors()
    
    # Aggregate by district
    district_counts = predictions.groupby(['district', 'prediction']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    district_counts.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=[colors.get(col, '#808080') for col in district_counts.columns]
    )
    
    ax.set_xlabel('District')
    ax.set_ylabel('Number of Booths/Wards')
    ax.set_title('Predicted Election Outcomes by District')
    ax.legend(title='Party', bbox_to_anchor=(1.02, 1))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_prediction_report(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: Optional[np.ndarray],
    class_names: List[str],
    save_dir: str
) -> None:
    """
    Create a comprehensive prediction report with multiple visualizations.
    
    Args:
        predictions: Predicted class indices
        probabilities: Prediction probabilities
        true_labels: True labels (optional, for evaluation)
        class_names: Names of classes
        save_dir: Directory to save report files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prediction distribution
    plot_prediction_distribution(
        probabilities,
        class_names,
        save_path=os.path.join(save_dir, 'prediction_distribution.png')
    )
    
    # If true labels available, create evaluation plots
    if true_labels is not None:
        # Confusion matrix
        plot_confusion_matrix(
            true_labels,
            predictions,
            class_names,
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Calculate metrics
        from sklearn.metrics import classification_report
        report = classification_report(
            true_labels,
            predictions,
            target_names=class_names,
            output_dict=True
        )
        
        # Save metrics
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.to_csv(os.path.join(save_dir, 'classification_metrics.csv'))
    
    # Summary statistics
    summary = {
        'Total Predictions': len(predictions),
        'Prediction Counts': dict(zip(
            class_names,
            [int((predictions == i).sum()) for i in range(len(class_names))]
        )),
        'Average Confidence': float(probabilities.max(axis=1).mean()),
        'High Confidence (>0.7)': int((probabilities.max(axis=1) > 0.7).sum()),
        'Low Confidence (<0.5)': int((probabilities.max(axis=1) < 0.5).sum())
    }
    
    # Save summary
    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write("Kerala Election Prediction Summary\n")
        f.write("=" * 40 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Report saved to {save_dir}")


# Testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Sample predictions
    n_samples = 100
    n_classes = 4
    class_names = ['LDF', 'UDF', 'NDA', 'OTHERS']
    
    probabilities = np.random.dirichlet(np.ones(n_classes), n_samples)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = np.random.choice(n_classes, n_samples, p=[0.35, 0.40, 0.15, 0.10])
    
    # Create plots
    fig1 = plot_prediction_distribution(probabilities, class_names)
    fig2 = plot_confusion_matrix(true_labels, predictions, class_names)
    
    # Feature importance
    importance = {
        'Sentiment': 0.45,
        'Historical': 0.35,
        'Demographic': 0.20
    }
    fig3 = plot_feature_importance(importance)
    
    plt.show()
    print("Visualization test complete!")
