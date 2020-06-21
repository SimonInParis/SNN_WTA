**SNN_WTA**

Uses:

`Python 3.6.9`

`bindsnet 0.2.7`



A modified WTA learning scheme, trained on MNIST, based on first firing neurons:

structure: input layer > hidden layer > locally recurrent output layer

STDP (PostPre) learning.

Average first spike for output neurons is used to determine MNIST digit.


Best accuracy is about 85%, but decreases after 20% of MNIST training as of now.
