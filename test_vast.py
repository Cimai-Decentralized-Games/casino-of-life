from casino_of_life.src.gpu_bridge import VastTrainingBridge

# Create test script
vast_bridge = VastTrainingBridge({
    'host': '70.69.205.56',
    'port': 57258,
    'workspace': '/workspace/casino-of-life'
})

try:
    # Test SSH connection
    print("Testing SSH connection...")
    vast_bridge.connect()
    print("SSH connection successful!")
    
    # Test command execution
    print("\nTesting command execution...")
    vast_bridge.ssh.exec_command('ls /workspace/casino-of-life')
    print("Command execution successful!")
    
except Exception as e:
    print(f"Connection failed: {str(e)}")
finally:
    vast_bridge.close()
