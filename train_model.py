"""
Re-train the Patient Digital Twin model (the SVM from the notebook).

    python train_model.py

The app also does this automatically the first time it starts.
"""
from core.tabular import train_and_save

if __name__ == "__main__":
    train_and_save(verbose=True)
