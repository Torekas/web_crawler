import subprocess
import sys
import os
from datetime import datetime

def run_command(command, log_file_handle):
    """
    Runs a command, streaming output to both console and a log file.
    """
    cmd_str = ' '.join(command)
    header = f"[{datetime.now().strftime('%H:%M:%S')}] Executing: {cmd_str}\n"
    
    # Write header to console and log
    print(f"\n{'-'*60}")
    print(header.strip())
    print(f"{'-'*60}")
    log_file_handle.write(f"\n{'-'*60}\n{header}{'-'*60}\n")
    log_file_handle.flush()

    try:
        # Popen allows us to read stdout in real-time
        # stdout=subprocess.PIPE captures output
        # stderr=subprocess.STDOUT merges errors into the standard output stream
        with subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            encoding='utf-8'
        ) as proc:
            
            # Read line by line as they occur
            for line in proc.stdout:
                sys.stdout.write(line) # Print to console
                log_file_handle.write(line) # Write to file
            
            proc.wait() # Wait for process to finish

            if proc.returncode != 0:
                error_msg = f"\nError: Command exited with code {proc.returncode}"
                print(error_msg)
                log_file_handle.write(error_msg + "\n")
                sys.exit(proc.returncode)

    except Exception as e:
        print(f"Failed to execute command: {e}")
        log_file_handle.write(f"Failed to execute command: {e}\n")
        sys.exit(1)

# --- Configuration ---

# 1. Crawl Command
crawl_cmd = [
    sys.executable, "-m", "src.main", "crawl",
    "--max-pages", "300",
    "--depth", "5",
    "--concurrency", "6",
    "--delay", "0.8",
    "--judge-llm", "ollama",
    "--judge-model", "mixtral:8x7b",
    "--output", "data/pages.jsonl"
]

# 2. Index Command
index_cmd = [
    sys.executable, "-m", "src.main", "index",
    "--pages", "data/pages.jsonl",
    "--index", "data/index.pkl.gz",
    "--model", "sentence-transformers/all-MiniLM-L6-v2"
]

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate a unique filename based on time: e.g., "logs/run_2023-10-27_14-30-00.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join("logs", f"run_{timestamp}.log")

    print(f"Logging output to: {log_filename}")

    # Open the log file once and pass it to the runner
    with open(log_filename, "w", encoding="utf-8") as log_file:
        run_command(crawl_cmd, log_file)
        run_command(index_cmd, log_file)
        
        success_msg = f"\n[{datetime.now().strftime('%H:%M:%S')}] Pipeline finished successfully.\n"
        print(success_msg)
        log_file.write(success_msg)