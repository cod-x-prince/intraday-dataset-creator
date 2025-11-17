# run_worker.py
import subprocess

def main():
    """
    Starts the Celery worker using a subprocess command.
    This is a robust way to launch the worker on Windows.
    """
    print("--- LAUNCHER: Starting Celery worker... ---")
    
    # The command we know works
    command = "celery -A worker.app worker --loglevel=info -P eventlet"
    
    # Run the command. This will take over the current terminal.
    # Use Ctrl+C to stop it.
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()