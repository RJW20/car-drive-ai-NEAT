# Car Drive AI: NEAT
An application of my implementation of [NEAT](https://github.com/RJW20/NEAT) to a simple car driving around a track.

## Configuration

### What the Car can see:
The Car uses a LIDAR like system to determine the distance to the Track from the Car along the line straight ahead, along the lines $-+$ $\pi/6$, $\pi/3$, $\pi/2$ emitted from the front left/right of the Car, along the lines $-+$ $3\pi/4$ emitted from the back left/right of the Car, and straight behind. The furthest the Car can see is three times its own length in all these directions, and the values passed to the Neural Network use this as the normalisation. The visual below shows this in action:

![lidar](https://github.com/RJW20/car-drive-ai-NEAT/assets/99192767/bdfe8b88-87a8-42b3-ba98-7d79c9ac207b)

Three more values are also passed to the Neural Network. These are:
- The Car's speed.
- The difference in angle between the Car's velocity and the direction its facing (essentially drift angle).
- The current turning angle.

### What the Car can do:
At every frame the Car makes two decisions. Firstly it can choose to either do nothing, accelerate or brake. Then it also sets the turn angle of the wheels by using a continuous map from one node to [ $-\pi/4$, $\pi/4$].

### Neural Network Structure:
The algorithm starts off with networks that have 13 input nodes, 1 bias node, 4 output nodes with sigmoid activation, and one random connection. The output node of the first three with highest activation chooses whether to do nothing, accelerate or brake and the fourth sets the turn angle of the front wheels.

### Fitness Function:
Two fitness functions are used in the training. First, the fitness of the Car is simply $f_c = g$ where $g$ is just the number of Track gates passed through forwards (i.e. how far the Car has travelled around the Track). Then, once one lap has been completed a different function $f_c \propto \dfrac{g^2}{t}$ where $t$ is the time taken to go through the gates $g$ is used to encourage the AI to drive faster (the constant of proportionality enables the Species to not go stale incorrectly when switching functions).

## Results
The AI gets pretty good at completing fast laps considering it is only given local information and doesn't have knowledge about the overall Track layout so it can't plan ahead to find what would be the optimal racing line. Below are a few examples of such AIs completing the Tracks they were trained on in real time:

![t1](https://github.com/RJW20/car-drive-ai-NEAT/assets/99192767/a39969d7-414e-4851-99b8-df854371939e)
![t2](https://github.com/RJW20/car-drive-ai-NEAT/assets/99192767/23b918f3-619d-47e2-9dfe-4a1b1bf8c4e8)
![t3](https://github.com/RJW20/car-drive-ai-NEAT/assets/99192767/04fd9a60-062a-4317-b210-ada8fde4f136)

## If you want to run it yourself

### Basic Requirements:
1. [Python](https://www.python.org/downloads/).
2. [Poetry](https://python-poetry.org/docs/) for ease of installing the dependencies.

### Getting Started:
1. Clone or download the repo `git clone https://github.com/RJW20/car-drive-ai-NEAT.git`.
2. Download the submodules `git submodule update --init`.
3. Set up the virtual environment `poetry install`.

### Running the Algorithm:
1. Create a track with <code>poetry run *method* *plane_width* *plane_height* *track_name*</code> using any of the methods described [here](https://github.com/RJW20/car-drive-app#tracks).
2. Change any settings you want in `car_drive_ai/settings.py`. For more information on what they control see [here](https://github.com/RJW20/NEAT/blob/main/README.md#configuring-the-settings). Make sure to include your `track_name` in `simulation_settings`.
3. Run the algorithm `poetry run main`.
5. View the saved playback with `poetry run playback`. You can change the generation shown with the left/right arrow keys, the species shown with the up/down arrows (or the spacebar for all species in the current generation).
