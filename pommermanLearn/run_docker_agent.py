"""Implementation of a simple deterministic agent using Docker."""

from agents.docker_agent import DockerAgent

def main():
    """
    Inits and runs a Docker Agent
    """
    model_dir="./data/models/communicate_2_99_1.pth" # Agent 1
    # model_dir="./data/models/communicate_2_99_2.pth" # Agent 2

    agent = DockerAgent(model_dir)
    agent.run()


if __name__ == "__main__":
    main()
