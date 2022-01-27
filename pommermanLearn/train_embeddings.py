import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import pommerman as pm
from pommerman.agents.simple_agent import SimpleAgent

from embeddings import PommerLinearAutoencoder
from embeddings import PommerConvAutoencoder
from util.data import transform_observation

# General
log_level=logging.INFO
train=True
evaluate=True

# Dataset
dataset_path = None # If set the dataset will be loaded from disk *instead* of generated
num_games   = 5
dataset_type = 'flattened' # One of 'planes' or 'flattened'
train_split = 0.8
val_split   = 1-train_split

# Training
num_epochs  = 10000
learning_rate=0.001

def play_game(env):
    """
    Play a game in the given environment and return the collected
    observations
    """
    observations=[]
    observation=env.reset()
    done=False

    while not done:
        act=env.act(observation)
        observation, rwd, done, _ = env.step(act)
        observations.append(observation)
        #env.render()
    env.close()

    return observations

def generate_dataset(num_games, flatten=True):
    X=[]
    for i in range(num_games):
        # Generate data by playing games with simple agents
        agent_list=[
            SimpleAgent(),
            SimpleAgent(),
            SimpleAgent(),
            SimpleAgent()
        ]

        env = pm.make("PommeRadioCompetition-v2", agent_list)
        observations=play_game(env)
        for observation in observations:
            for view in observation:
                planes = transform_observation(view, p_obs=True, centralized=True)
                if flatten:
                    x = planes.flatten()
                else:
                    x = planes
                X.append(np.array(x, dtype=np.float32))
        logging.info(f"Finished game {i+1}/{num_games} in {len(observations)} steps")
    return X
    
def train_model(model, X_train, X_val, epochs=1, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        for x in X_train:
            optimizer.zero_grad()
            y = model(x.unsqueeze(0)).squeeze()
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(X_train)
        train_loss.append(loss)

        path="./data/models/embeddings_net_pobs-last.pth"
        if loss <= min(train_loss):
            path="./data/models/embeddings_net_pobs-best.pth"
        logging.info(f"Saving model to {path}")
        save_model(model, path)
        
        threshold, accuracy, precision, recall = evaluate_model(model, X_val, threshold=0.5)
        logging.info(f"Finished epoch {epoch+1}/{epochs}; Loss: {loss}; Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}")#; Accuracy {accuracy_score(x_val, y_val)} Precision {precision_score(x_val, y_val)}; Recall {recall_score(x_val, y_val)}")
    return train_loss
            
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

def setup_logger(log_level=logging.INFO):
    """
    Setup the global logger

    :param log_level: The minimum log level to display. Can be one of
        pythons built-in levels of the logging module.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)

def load_model(path: str, model: nn.Module) -> nn.Module:
   model.load_state_dict(torch.load(path))
   model.eval() 
   return model

def preprocess_data(x):
    x = torch.tensor(x, device=get_device())
    return x

def postprocess_data(y):
    y = np.reshape(y.detach().numpy(), (-1, 9, 9))
    return y

def evaluate_model(model, data, threshold=0.5):
    recall=[]
    precision=[]
    accuracy=[]
    for x in data:
        reconstruction = model.forward(x.unsqueeze(0)).detach().numpy()

        expected = x.detach().numpy().flatten()
        actual = ((reconstruction>threshold)*1).flatten()

        accuracy.append(accuracy_score(expected, actual))
        precision.append(precision_score(expected, actual))
        recall.append(recall_score(expected, actual))
    return (threshold, sum(accuracy)/len(accuracy), sum(precision)/len(precision), sum(recall)/len(recall))

def test_thresholds(model, data, thresholds=[]):
    evaluations=[]
    for threshold in thresholds:
        evaluation=evaluate_model(model, data, threshold)
        logging.info(f"Threshold: {evaluation[0]}; Accuracy: {evaluation[1]}; Precision: {evaluation[2]}; Recall: {evaluation[3]}")
        evaluations.append(evaluation)
    return evaluations

setup_logger(log_level=log_level)


#model = PommerConvAutoencoder()
model = PommerLinearAutoencoder(1053)

data = None
if dataset_path is None:
    logging.info(f"Generating dataset from {num_games} games")
    if dataset_type == 'flattened':
        flatten = True
    else:
        flatten = False
    data = generate_dataset(num_games=num_games, flatten=flatten)
else:
    logging.info(f"Loading dataset from {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
logging.info(f"Proceeding with {len(data)} samples")

split=int(len(data)*train_split)
logging.info(f"Preprocessing train data ({split} items)")
X_train = preprocess_data(data[:split])

logging.info(f"Preprocessing validation data ({len(data)-split} items)")
X_val   = preprocess_data(data[split:])

if train:
    logging.info("Setting up model")
    model.to(get_device())

    logging.info("Starting training of model")
    loss=train_model(model, X_train, X_val, epochs=num_epochs, lr=learning_rate)

if evaluate:
    path="./data/models/embeddings_net_pobs-best.pth"
    logging.info(f"Loading model {path}")

    model = load_model(path, model)
    model.mode='both'

    logging.info("Evaluating with treshold of 0.5")
    threshold, accuracy, precision, recall = evaluate_model(model, X_val, 0.5)
    logging.info(f"Accuracy: {accuracy}; Precision: {precision}; Recall: {recall}")

    logging.info("Evaluating tresholds in range of 0 to 1.0 in steps of 0.01")
    evaluations=test_thresholds(model, X_val, thresholds=[t/100 for t in range(101)])
    f = open("eval.csv", "w")
    f.write(f"threshold;accuracy;precision;recall")
    for evaluation in evaluations:
        f.write(f"\n{evaluation[0]};{evaluation[1]};{evaluation[2]};{evaluation[3]}")