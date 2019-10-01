# N of 1 Trial Study

## Effects of Inhibition on Trial Signals in 'numberStarred' Game

In this notebook we look on the effects of inhibitory control on executive function. 

At `t=0` a visual stimulus appears on the screen. As the response, a participant (person holding the phone) is required to tilt the phone in a specific direction (either left, right, up, or down). Once the roll or pitch angle reaches 0.5 radians (`t=T>0`), the movement is completed. `T` is the event time registered by the app. The rotation along the third axis (yaw) *does not* affect the outcome of the task, hence it is neglected.

When the cursor `star` appears on the screen, the commands are reversed along the axis of the target. (Left becomes right, right becomes left; up becomes down, down becomes up.) This is response inhibition.

In order to establish a statistical difference between inhibitory and non-inhibitory responses, we extract a number of features from the mobile phone orientation data. The features extracted are: reaction time, L1 norms, maximal deviation, number of turning points, and sample entropy.

