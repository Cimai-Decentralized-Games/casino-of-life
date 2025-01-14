# 🎮 Casino of Life

A revolutionary framework for training AI agents in retro fighting games using natural language interactions. Casino of Life combines reinforcement learning with natural language processing to create an intuitive interface for training game-playing AI agents.

## 🌟 Features

### Natural Language Training Interface
- Train AI agents using natural conversations
- Explain strategies in plain English
- Get real-time feedback on training progress
- Interactive chat with CaballoLoko, your AI training assistant

### Supported Games
- Mortal Kombat II (Genesis)
- Street Fighter II (Coming Soon)
- More fighting games to be added

### Advanced Training Capabilities
- Multiple training strategies (Aggressive, Defensive, Balanced)
- Various learning policies (PPO, A2C, DQN, MLP)
- Custom reward functions
- Save and load training states
- Real-time training visualization

## 🚀 Quick Start

### Installation
```bash
pip install casino-of-life
```

### Basic Usage
```python
from casino_of_life.environment import CasinoFightingEnv
from casino_of_life.agents import FightingAgent

# Create environment
env = CasinoFightingEnv(
    game='MortalKombatII-Genesis',
    character='liu-kang',
    strategy='aggressive'
)

# Initialize agent
agent = FightingAgent(
    env=env,
    policy='PPO',
    learning_rate=0.0003
)

# Start training with natural language guidance
agent.train(
    message="Train Liu Kang to be aggressive with jump kicks and fireballs",
    timesteps=100000
)
```

## 🎯 Use Cases

### Game Developers
- Test game balance
- Create sophisticated AI opponents
- Generate training data for game testing

### AI Researchers
- Experiment with reinforcement learning in complex environments
- Study human-AI interaction through natural language
- Develop and test new training strategies

### Gaming Community
- Create custom AI training scenarios
- Share and compare training results
- Contribute to the evolution of game AI and the Casino of Life framework. 

## 🛠 Advanced Features

### Custom Training Scenarios
```python
from casino_of_life.scenarios import create_scenario

scenario = create_scenario(
    name="tournament",
    difficulty="hard",
    opponent="random",
    rounds=3
)
```

### Training State Management
```python
# Save training progress
agent.save_state("liu_kang_aggressive_v1")

# Load existing training
agent.load_state("liu_kang_aggressive_v1")
```

### Real-time Metrics
```python
from casino_of_life.metrics import TrainingMonitor

monitor = TrainingMonitor(agent)
monitor.start_tracking()
```

## 📊 Web Interface

Access the training interface at [https://casino.cimai.biz](https://casino.cimai.biz)
- Interactive chat with CaballoLoko
- Real-time training visualization
- Model management interface
- Community features

## �� Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/Cimai-Decentralized-Games/casino-of-life.git
cd casino-of-life
pip install -r requirements.txt
```

## 📚 Documentation

Full documentation available at [https://docs.cimai.biz](https://docs.cimai.biz)
- API Reference
- Training Guides
- Example Projects
- Best Practices

## 🔗 Links
- [Website](https://cimai.biz)
- [Documentation](https://docs.cimai.biz)
- [Discord Community](https://discord.gg/cimai)
- [Blog](https://blog.cimai.biz)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [G4F](https://github.com/xtekky/gpt4free) for GPT models and other Providers
- [Stable-Retro](https://github.com/Farama-Foundation/stable-retro) for game emulation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL implementations
- The fighting game community for inspiration and support



---

Made with ❤️ by Cimai Decentralized Games
