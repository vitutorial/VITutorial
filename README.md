# VITutorial
This repository stores slides for a tutorial on variational inference for NLP audiences. Let us know if you are interested in hosting the tutorial.

# News

* We are now working on mxnet code for variational autoencoders. Soon this will be turned into a jupyter notebook.
* The tutorial has already been presented in Berlin, where it was given in 2 sessions. The focus was on Deep Generative Models ([Module 6](https://github.com/philschulz/VITutorial/blob/master/modules/M6_DeepGenerativeModels/M6_DeepGenerativeModels.pdf)). The next stop will be Dublin.

# Tour
Below are confirmed venues where we will give the tutorial.
* Melbourne University
* Dublin City University

# Latex Dependencies
To compile the slides, latex needs to have access to the [bayesnet tikz library](https://github.com/jluttine/tikz-bayesnet).

# Python Code

## Usage

TBD

## Dependencies

To train and evaluate the model you need mxnet. For more detailed info
see [here](https://mxnet.incubator.apache.org/get_started/install.html).

```
pip install mxnet # cpu installation
pip install mxnet-cu80 # gpu installation with cuda 8
```

If you also want to visualize the digits, you need matplotlib.
```
pip install matplotlib
```
Now if you run the code, python might tell you that you are missing ```_tkinter```. This is bad news as it means 
that you'll have to rebuild python. First, you have to
```
sudo apt-get install tk-dev
```
and after rebuilding python should be configured for Tk.
