import os
import pickle

import torch
import yaml
from experience_trainer.model.model import ActorCriticModel

from model.model import ActorCritic
from player.player import Player

if __name__ == "__main__":
    with open("config.yaml", 'r', encoding='utf-8') as conf:
        conf = yaml.safe_load(conf)

        task_file = open(conf['task_configuration'],'rb')
        task = yaml.safe_load(task_file)

        action_map_file = open(conf['action_map'], 'rb')
        action_map = pickle.load(action_map_file)

        model = ActorCriticModel()


        checkpoint = torch.load(conf['checkpoint_path'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()


        # model.load_state_dict(torch.load(conf['model_path'])
        #
        player = Player(task,model,action_map,'board')
        player.start()

