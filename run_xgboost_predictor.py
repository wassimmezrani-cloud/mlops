#!/usr/bin/env python3
"""
XGBoost Predictor Runner avec Virtual Environment
Exécute le XGBoost predictor avec l'environnement virtuel
"""

import subprocess
import sys
import os

def run_xgboost_predictor():
    """Run XGBoost predictor with virtual environment"""
    
    # Check if virtual environment exists
    venv_path = "./mlops_env"
    if not os.path.exists(venv_path):
        print("Error: Virtual environment not found. Please create it first.")
        return False
    
    # Command to run with virtual environment
    cmd = [
        "bash", "-c",
        "source mlops_env/bin/activate && python3 notebooks/xgboost_predictor.py"
    ]
    
    print("Starting XGBoost Predictor with virtual environment...")
    print("XGBoost 2.0.3 + MLflow 3.3.2 available")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=600)
        
        if result.returncode == 0:
            print("\n✅ XGBoost Predictor completed successfully!")
            return True
        else:
            print(f"\n❌ XGBoost Predictor failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ XGBoost Predictor timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"\n💥 Error running XGBoost Predictor: {e}")
        return False

if __name__ == "__main__":
    success = run_xgboost_predictor()
    sys.exit(0 if success else 1)