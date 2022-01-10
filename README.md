# The Introspective Agent:
## Interdependence of Strategy, Physiology, and Sensing for Embodied Agents

This is the code corresponding to [The Introspective Agent: Interdependence of Strategy, Physiology, and Sensing for Embodied Agents](https://arxiv.org/pdf/2201.00411.pdf) by Sarah Pratt, Luca Weihs, and Ali Farhadi.

## Abstract:

The last few years have witnessed substantial progress in the field of embodied AI where artificial agents, mirroring biological counterparts, are now able to learn from interaction to accomplish complex tasks. Despite this success, biological organisms still hold one large advantage over these simulated agents: adaptation. While both living and simulated agents make decisions to achieve goals (strategy), biological organisms have evolved to understand their environment (sensing) and respond to it (physiology). The net gain of these factors depends on the environment, and organisms have adapted accordingly. For example, in a low vision aquatic environment some fish have evolved specific neurons which offer a predictable, but incredibly rapid, strategy to escape from predators. Mammals have lost these reactive systems, but they have a much larger fields of view and brain circuitry capable of understanding many future possibilities. While traditional embodied agents manipulate an environment to best achieve a goal, we argue for an introspective agent, which considers its own abilities in the context of its environment. We show that different environments yield vastly different optimal designs, and increasing long-term planning is often far less beneficial than other improvements, such as increased physical ability. We present these findings to broaden the definition of improvement in embodied AI passed increasingly complex models. Just as in nature, we hope to reframe strategy as one tool, among many, to succeed in an environment

## Code

### Training
To train the predator and prey, run the following command:

```
python train.py --planning PLANNING --speed SPEED --vision VISION
```

Planning has the options of ['low', 'mid', 'high']
Speed has the options of ['veryslow', 'slow', 'average', 'fast', 'veryfast']
Vision has the options of ['short', 'medium', 'long']

So an example command looks like this:

```
python train.py --planning high --speed average --vision long
```

Prey and Predator weights will save every 1000 gradient updates under a folder of the form log/planning_high_vision_long_speed_average (as an example corresponding to the example train command)


### Evaluation
Our evaluation metric is the number of times that the predator is able to catch the prey in 10,000 steps. The location of the prey is randomly reset after it is caught by the predator (or after 400 steps to avoid outlier episodes).

To evaluate the training run, use the command:

```
python eval.py --prey-weights ./PATH/TO/PREY/WEIGHTS --predator-weights ./PATH/TO/PREY/WEIGHTS --speed fast --vision short --planning high
```

which will output a string of the form:

```
Number of prey caught in 10,000 steps is NUMBER
```

To visualize a video of the evaluation run, use the flag --video

Prerequisite packages can be found in requirements.txt
