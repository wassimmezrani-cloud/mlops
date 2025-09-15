#!/usr/bin/env python3
"""
Q-Learning Optimizer Runner
Execute Q-Learning with virtual environment isolation
Same pattern as XGBoost runner for consistency
"""

import subprocess
import sys
import os

def main():
    """Execute Q-Learning Optimizer with proper environment"""
    print("Starting Q-Learning Optimizer with virtual environment...")
    
    venv_path = "./mlops_env"
    script_path = "notebooks/qlearning_optimizer.py"
    
    # Check virtual environment
    if not os.path.exists(venv_path):
        print("ERROR: Virtual environment not found at ./mlops_env")
        print("Please run XGBoost setup first to create the environment")
        return 1
    
    # Check script
    if not os.path.exists(script_path):
        print(f"ERROR: Q-Learning script not found at {script_path}")
        return 1
    
    # Check dependencies available
    activate_script = f"{venv_path}/bin/activate"
    if not os.path.exists(activate_script):
        print("ERROR: Virtual environment activation script not found")
        return 1
    
    try:
        # Execute with virtual environment
        cmd = f"source {activate_script} && python3 {script_path}"
        result = subprocess.run(cmd, shell=True, executable="/bin/bash", 
                              capture_output=True, text=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            print("\n✅ Q-Learning Optimizer completed successfully!")
        else:
            print(f"\n❌ Q-Learning Optimizer failed with return code: {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        print(f"ERROR: Failed to execute Q-Learning Optimizer: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())