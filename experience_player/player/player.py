import logging
import sys

import pyautogui
import torch
from torchvision import transforms


fmt = '[%(asctime)-15s] [%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


class Player:
    def __init__(self, task, model, action_map, state_sense):
        self.task = task
        self.model = model
        self.action_map = action_map
        self.state_sense = state_sense
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to the input size of ResNet
            transforms.ToTensor(),
        ])
        logger.info('Player initialized.')

    def get_display_state(self):
        logger.debug('Taking a screenshot.')
        return self.transform(pyautogui.screenshot(region=self.task['senses'][self.state_sense]['location'])).unsqueeze(0).unsqueeze(2)

    def start(self):
        logger.info('Starting the Player.')
        previous_state = torch.rand([3,224,224])  # Initialize previous state as None

        while True:
            current_state = self.get_display_state()
            prediction = self.model(current_state)
            actor_output = prediction[1].squeeze()

            actions_priority = torch.argsort(actor_output, descending=True)
            action = self.action_map[actions_priority[0]].split('.')[1]
            logger.info(f'Performing action: {action}')
            pyautogui.press(action)

            previous_state = current_state
            current_state = self.get_display_state()
            index = 1
            while torch.allclose(previous_state,current_state) and index < len(self.action_map):
                action = self.action_map[actions_priority[index]].split('.')[1]
                logger.info(f'Retrying action due to state not changing: {action}')

                previous_state = current_state
                pyautogui.press(action)
                current_state = self.get_display_state()

                index += 1
            logger.debug('Moving to the next iteration.')

