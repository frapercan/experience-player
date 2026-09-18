# Experience Player

Inference and action module of the **Experience-Learning** project. Loads a
trained actor-critic checkpoint and plays the task (2048) through the computer
interface, the same way the recordings were captured.

## What it does

- Loads a trained model checkpoint and the action mapping produced during
  training.
- Captures the current board by screenshotting the configured region.
- Runs the model to rank actions, then presses the top key with `pyautogui`.
- Detects when the board did not change and retries the next-best action.

## Stack

PyTorch, torchvision (ResNet-18), pyautogui, PyYAML.

## Run

```bash
poetry install
# edit experience_player/config.yaml: action_map, task_configuration,
# checkpoint_path
python -m experience_player.main
```

## Part of the Experience-Learning pipeline

[recorder](https://github.com/frapercan/experience-recorder) →
[modeler](https://github.com/frapercan/experience-modeler) →
[trainer](https://github.com/frapercan/experience-trainer) → **player**, with the
[2048 environments](https://github.com/frapercan/2048-experience-learning) as
the task.
