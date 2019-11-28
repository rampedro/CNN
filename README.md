# CNN
Here the goal is to implement a Convolutional neural network Model of Machine learning and study the classification performance of a CNN. I will be training a neural network and I will use effective ways to avoid overfitting. The implementation are done in Python and Tensor-flow Library.


# Drop Outs 

Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

You can imagine that if neurons are randomly dropped out of the network during training, that other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.

The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.

# Flatten Layer 

if now: model.output_shape == (None, 64, 32, 32)
after model.add(Flatten())
then model.output_shape == (None, 64*32*32) 
