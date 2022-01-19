"""Implementation of a simple deterministic agent using Docker."""

from pommermanLearn.agents.docker_agent import DockerAgent

def main():
    """
    Inits and runs a Docker Agent
    """
    agent = DockerAgent()
    agent.run()


if __name__ == "__main__":
    main()
