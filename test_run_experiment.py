import subprocess
import sys

def main():
    print("Running a test of run_experiment.py for 5-node instances...")
    
    # Construct the command to run run_experiment.py with specific arguments
    command = [
        sys.executable,  # Ensures the script is run with the same python interpreter
        "run_experiment.py",
        "--n_nodes", "7",
        "--amount_of_instances", "1",
        "--time_limit", "30", # Short time limit for a quick test
        "--verbosity", "low",
        # use all three cut types for testing
        "--seq_cut_type", "dual", "enumerative", "dual+enumerative",
        # Default values for other parameters will be used if not specified here
        # E.g., --r_mean, --sigma, --model_type, --decomposition, --extended, --seq_cut_type
    ]
    
    try:
        # Execute the command
        subprocess.run(command, check=True, text=True, capture_output=False)
        print("\nTest run completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during test run: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"\nError: '{sys.executable}' or 'run_experiment.py' not found.")
        print("Please ensure Python is correctly installed and 'run_experiment.py' is in the current directory.")

if __name__ == "__main__":
    main()

