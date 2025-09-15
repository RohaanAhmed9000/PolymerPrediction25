import pandas as pd

def save_predictions_csv(predictions, filename="predictions.csv", ids=None):
    """Save predictions to CSV file"""
    
    
    if ids is None:
        ids = range(len(predictions))
    
    df = pd.DataFrame({
        'id': ids,
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1], 
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")

