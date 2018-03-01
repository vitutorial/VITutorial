This repository stores material for a *tutorial on variational inference for NLP audiences*. 

Want to host our tutorial at your location? Contact one of us!
* [Philip Schulz](//philipschulz.org)
* [Wilker Aziz](//wilkeraziz.github.io)


# Variational Inference and Deep Generative Models

Neural networks are taking NLP by storm. Yet they are mostly applied to fully supervised tasks. 
Many real-world NLP problems require unsupervised or semi-supervised models, however, because annotated data is hard to obtain. 
This is where generative models shine. 
Through the use of latent variables they can be applied in missing data settings. Furthermore they can complete missing entries in partially annotated data sets.

This tutorial is about how to use neural networks inside generative models, thus giving us Deep Generative Models (DGMs). 
The training method of choice for these models is variational inference (VI). 
We start out by introducing VI on a basic level. From there we turn to DGMs. 
We justify them theoretically and give concrete advise on how to implement them. For continuous latent variables, we review the variational autoencoder and use Gaussian reparametrisation to show how to sample latent values from it. 
We then turn to discrete latent variables for which no reparametrisation exists. 
Instead, we explain how to use the score-function or REINFORCE gradient estimator in those cases. 
We finish by explaining how to combine continuous and discrete variables in semi-supervised modelling problems.

# News

* The [tutorial code](code/vae_notebook.ipynb) is now available! The user still needs to complete the TODOs in order for the code to run.
Make sure to follow the instructions and read the comments carefully. Also check out the links to the MXNet documention.

# Tour

**Upcoming**

Below are confirmed venues and dates (if available) for future presentations of the tutorial. Please contact us
if you interested in hosting the tutorial.

* Uva-ILLC, Amsterdam: March 22, 2018 (room TBA)
* ACL 2018, Melbourne: July 15th, 2018

**Past**
* Monash University
  1. Basics of Variational Inference: Thu, 16-11-217, 10am-11:30am
  2. Deep Generative Models: Thu, 16-11-2017, 2:30pm-4pm
* Melbourne University
  1. Basics of Variational Inference: Tue, 31-10-2017, Doug McDonell Building, room 8.03, 2:00pm-3:15pm
  2. Deep Generative Models: Thu, 02-11-2017, Doug McDonell Building, room 8.03, 2:15pm-3:30pm
  3. Coding Tutorial: Tue, 07-11-2017, Doug McDonell Building, room 8.03, 2:00pm-3:15pm
* Berlin, July 26-27 2017

# Latex Dependencies
To compile the slides, latex needs to have access to the [bayesnet tikz library](https://github.com/jluttine/tikz-bayesnet).

# Python Code

While we strive to update our code base with new and more complex models, the Gaussian VAE is at the heart of the tutorial.
See [here](code/gaussian_vae.pdf.gv.pdf) for what the computation graph of such a model looks like. (No worries, it's actually
pretty straightforward to implement.) 

## Dependencies
**Framework**: Our code uses MXNet which is a scalable machine learning library that is under active development.
For more details on how to install MXNet see [here](https://mxnet.incubator.apache.org/get_started/install.html).

To run the tutorial code and notebook, we recommend that you setup a virtual environment. Your Python version
should be 3.5 or higher.

**Warning**: If you are using linux and Python3.6 or higher you need to run `sudo apt-get install libssl-dev` before
building Python. Otherwise, there is a chance that your virtualenv will not be able to download packages.
```
virtualenv -p python3.5 vi-tutorial-env
source vi-tutorial-env/bin/activate
pip install mxnet # cpu installation
pip install mxnet-cu80 # gpu installation with cuda 8
pip install jupyter matplotlib
```
**Issues with matplotlib:** If you are using linux and building python from source, pyplot might not work for you 
because it's missing tkinter (`_tkinter` module). In that case run `sudo apt-get install tk-dev` and rebuild python.

## Usage

Once you have executed the above commands, open a notebook with `jupyter notebook`. Then use your browser to navigate
to the notebook. The notebook file is: `<path to repo>/VITutorial/code/vae_notebook.ipynb`. Make sure to have activated
the `vi-tutorial-env` environment before starting the notebook.
