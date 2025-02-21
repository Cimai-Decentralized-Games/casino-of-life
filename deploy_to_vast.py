import os
import subprocess

def deploy_to_vast():
    host = "70.69.205.56"
    port = "57258"
    remote_path = "/workspace/casino-of-life"
    
    # Sync package files
    subprocess.run([
        "rsync", 
        "-avz", 
        "-e", f"ssh -p {port}",
        "./casino-of-life/",
        f"root@{host}:{remote_path}"
    ])
    
    # Install requirements
    subprocess.run([
        "ssh",
        "-p", port,
        f"root@{host}",
        f"cd {remote_path} && pip install -r requirements.txt"
    ])

if __name__ == "__main__":
    deploy_to_vast()