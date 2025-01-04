import paramiko
import json
from pathlib import Path
from casino_of_life.src.vast_config import VAST_INSTANCE
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VastTrainingBridge:
    def __init__(self, config=None):
        logger.debug(f"Initializing VastTrainingBridge with config: {config}")
        self.config = config or VAST_INSTANCE
        logger.debug(f"Using config: {self.config}")
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    def connect(self):
        try:
            logger.debug("Attempting SSH connection...")
            logger.debug(f"Config keys available: {self.config.keys()}")
            
            if 'ssh_key' not in self.config:
                logger.error("No ssh_key found in config!")
                logger.debug(f"VAST_INSTANCE contains: {VAST_INSTANCE}")
                raise KeyError("ssh_key not found in config")
            
            # Write the SSH key to a temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as key_file:
                logger.debug("Writing SSH key to temporary file...")
                key_file.write(self.config['ssh_key'])
                key_path = key_file.name
            
            try:
                logger.debug(f"Connecting to {self.config['host']}:{self.config['port']}")
                self.ssh.connect(
                    self.config['host'],
                    port=self.config['port'],
                    username='root',
                    key_filename=key_path
                )
                logger.debug("SSH connection successful!")
            finally:
                os.unlink(key_path)  # Clean up the temporary file
                
        except Exception as e:
            logger.exception(f"SSH connection failed with error: {str(e)}")
            raise
    
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
    
    def get_status(self, instance_id=None):
        self.connect()
        command = f"cat {self.config['workspace']}/training_status.json"
        stdin, stdout, stderr = self.ssh.exec_command(command)
        return json.loads(stdout.read().decode())
    
    def close(self):
        self.ssh.close()
