import paramiko
import json
from pathlib import Path

class VastTrainingBridge:
    def __init__(self, config_path="src/vast_config.py"):
        self.config = self._load_config(config_path)
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    def connect(self):
        self.ssh.connect(
            self.config['host'],
            port=self.config['port'],
            username='root'
        )
    
    async def start_training(self, training_params):
        self.connect()
        
        # Save training params
        params_path = f"{self.config['workspace']}/current_training.json"
        stdin, stdout, stderr = self.ssh.exec_command(
            f"echo '{json.dumps(training_params)}' > {params_path}"
        )
        
        # Start training
        command = f"cd {self.config['workspace']} && python -m casino_of_life.train"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        
        return {
            'status': 'started',
            'output': stdout.read().decode()
        }
    
    def get_status(self):
        self.connect()
        command = f"cat {self.config['workspace']}/training_status.json"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        return json.loads(stdout.read().decode())
    
    def close(self):
        self.ssh.close()
