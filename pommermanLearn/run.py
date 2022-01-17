"""Implementation of a simple deterministic agent using Docker."""

from agents.deployment_agent import DeploymentAgent

def main():
    '''Inits and runs a Docker Agent'''
    agent = DeploymentAgent()
    agent.run()


if __name__ == "__main__":
    main()
