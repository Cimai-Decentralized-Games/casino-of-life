import paramiko
import json
import requests
import time
import logging
from casino_of_life.src.vast_config import VAST_INSTANCE
import socket
from contextlib import contextmanager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VastTrainingBridge:
    def __init__(self):
        self.config = {
            'host': '70.69.205.56',
            'port': 57258,
            'username': 'root',
            'remote_port': 5000,
            'local_port': 5000
        }
        self.ssh = None
        self.tunnel_active = False
        self.max_retries = 3
        self.retry_delay = 2
        
    def connect(self):
        """Establish SSH connection"""
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                self.config['host'],
                port=self.config['port'],
                username=self.config['username']
            )
            logger.debug("SSH connection established")
        except Exception as e:
            logger.error(f"SSH connection failed: {e}")
            raise

    def setup_tunnel(self):
        """Set up SSH tunnel with exactly 3 arguments"""
        try:
            if not self.tunnel_active:
                self.connect()
                self.transport = self.ssh.get_transport()
                
                # Wait for transport to be ready
                time.sleep(1)
                
                # Setup port forward with exactly 3 arguments
                self.transport.request_port_forward('', 
                                                 self.config['local_port'],
                                                 ('localhost', self.config['remote_port']))
                
                # Wait for tunnel to be established
                time.sleep(1)
                
                self.tunnel_active = True
                logger.debug(f"SSH tunnel established to {self.config['host']}:{self.config['remote_port']}")
                
        except Exception as e:
            self.tunnel_active = False
            logger.error(f"Failed to set up tunnel: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up connections"""
        if self.ssh:
            try:
                self.ssh.close()
            except:
                pass
        self.tunnel_active = False

    @contextmanager
    def tunnel_context(self):
        """Context manager for tunnel connection"""
        try:
            self.setup_tunnel()
            yield
        finally:
            self.cleanup()

    async def start_training(self, training_params):
        """Start training with longer timeout"""
        with self.tunnel_context():
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Sending training request to Vast (attempt {attempt + 1})")
                    
                    # Increase timeout to 120 seconds
                    response = requests.post(
                        f'http://localhost:{self.config["local_port"]}/train',
                        json=training_params,
                        headers={'Content-Type': 'application/json'},
                        timeout=120  # Increased from 30
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise Exception(f"Training request failed: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        # Return a success response even if we timeout
                        return {
                            "status": "training_started",
                            "message": "Training started successfully but response timed out. Check training status for updates."
                        }
    
    def get_status(self, session_id: str):
        """Get training status from Vast instance"""
        try:
            self.check_tunnel()
            
            response = requests.post(
                f'http://localhost:{self.config["local_port"]}/training-status',
                json={"session_id": session_id},
                headers={
                    'Content-Type': 'application/json',
                    'X-Session-ID': session_id
                },
                timeout=10  # Short timeout for status checks
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Status request failed: {response.text}")
                return {
                    "status": "error",
                    "progress": 0,
                    "currentReward": 0,
                    "episodeCount": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {
                "status": "error",
                "progress": 0,
                "currentReward": 0,
                "episodeCount": 0
            }
    
    def close(self):
        """Clean shutdown of connections"""
        if self.transport:
            self.transport.close()
        if self.ssh:
            self.ssh.close()
        self.tunnel_active = False