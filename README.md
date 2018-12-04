[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/philschulz/VITutorial/master)

This repository stores material for a *tutorial on variational inference for NLP audiences*. 

Want to host our tutorial at your location? [Contact](#contact) one of us!


**Useful links**

* General information
  * [The tutorial](#general)
  * [News](#news)
  * [Tour](#tour)
* Slides
  * [Variational inference](//github.com/philschulz/VITutorial/blob/master/modules/M1_Basics/M1_Basics.pdf)
  * [Continuous latent variables](//github.com/philschulz/VITutorial/blob/master/modules/M3a_DGMs_ContinuousLatentVariables/M3a_DGMs_ContinuousLatentVariables.pdf)
  * [Discrete latent variables](//github.com/philschulz/VITutorial/blob/master/modules/M3b_DGMs_DiscreteVariables/M3b_DGMs_DiscreteLatentVariables.pdf)
  * [Normalising Flows](//github.com/philschulz/VITutorial/blob/master/modules/M7_NormalisingFlows/M7_NormalisingFlows.pdf)
* Lecture notes
  * [Explaining reparameterisation](//github.com/philschulz/VITutorial/blob/master/modules/M3a_DGMs_ContinuousLatentVariables/ExplainingReparametrisationGradients.pdf)
  * Entropy and KL for exponential families
* Code
  * [Dependencies](#code)
  * [Jupyter notebooks](//github.com/philschulz/VITutorial/tree/master/code)
  

# <a name="general"> Variational Inference and Deep Generative Models

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

# <a name="news"> News
* A [new module on normalising flows](modules/M7_NormalisingFlows/M7_NormalisingFlows.pdf) has been added. Normalising flows are a way of learning distributions. Check it out to learn more. Soon we will also add module on ADVI, a black-box variational inference procedure.
* We have now tagged the version of the tutorial that we presented at ACL 2018 in Melbourne to simplify future reference. To get to that version, [click here](https://github.com/philschulz/VITutorial/tree/acl2018).
* We have added a [module on discrete latent variables](modules/M3b_DGMs_DiscreteVariables/M3b_DGMs_DiscreteLatentVariables.pdf).
This also led to a change in the module structure. The DGM part (M3) now consists of 2 interdependent modules. One presents
continuous latent variable models and the other discrete latent variable models.
* The [tutorial code](code/vae_notebook.ipynb) is now available! The user still needs to complete the TODOs in order for the code to run.
Make sure to follow the instructions and read the comments carefully. Also check out the links to the MXNet documention.

# <a name="tour"> Tour

**Upcoming**

Below are confirmed venues and dates (if available) for future presentations of the tutorial. Please contact us
if you interested in hosting the tutorial.

**Past**
* University of Heidelberg: Nov 29/30 2018
  1. Thursday, Nov 29: Basics of VI, continuous and discrete Deep Generative Models
  2. Friday, Nov 30: Automatic differentiation VI, normalising flows
* ACL 2018, Melbourne: July 15th, 2018
* Naver Labs, Grenoble, France: April 3 and April 6, 2018
  1. Deep Generative Models
* Uva-ILLC, Amsterdam: March 22, 2018
* Macquarie University Sydney: March 19-20, 2018
  1. Basics of Variational Inference
  2. Deep Generative Models
  3. Coding Tutorial
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

# <a name="code"> Python Code

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
virtualenv -p python3 vi-tutorial-env
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


# <a name="contact"> Contact

Want to host our tutorial? Have a suggestion? Contact one of us!

* [Philip](//philipschulz.org)
* [Wilker](//wilkeraziz.github.io)
