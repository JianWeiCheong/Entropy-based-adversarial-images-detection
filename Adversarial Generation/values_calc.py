"""
We calculate all the nonlinear measures here.
Unfortunately, we were unable to implement our own calculation algorithms.
Rather, we use the package 'nolds' to do it for us.

There are a few parameters the user should set based on what they want to calculate.
dataset : 'mnist', 'fashion', 'cifar10'
perturb : 'fgsm', 'deepfool', 'onepixel', 'sampen'
measure : 'sampen', 'frac', 'hurst', 'lyapr'

sampen  : sample entropy
frac    : fractal dimension
hurst   : hurst exponent
lyapr   : largest lyapunov exponent

The reason the user has to set these parameters themselves instead of simply
having the program calculate all measures for all perturbation for all dataset in one go,
is due to the long calculation time.
For example, calculating the fractal dimension of a single perturbation type for the
CIFAR10 dataset takes around 8 hours.

We expected almost no change in the nonlinear measures for onepixel perturbation,
this is because changing one pixel is not enough to affect the measures of the whole image.
Thus, we did not calculate the quantities for onepixel perturbation to save time.
"""

import keras
import numpy as np
from keras.datasets import cifar10, mnist, fashion_mnist

import nolds           # nolds is a package to calculate various nonlinear measures.
from tqdm import tqdm  # tqdm is a function that automatically generate a loading bar for loops.

import warnings
warnings.filterwarnings("ignore")


def main():
    # What you want to calculate.
    dataset = 'mnist'
    perturb = 'fgsm'
    measure = 'sampen'

    images, adversarial_images = load_images(dataset=dataset, perturb=perturb)
    imageCalc_data, advimageCalc_data = calc(images, adversarial_images, measure=measure)

    np.save('image_calc.npy', imageCalc_data)
    np.save('adv_calc.npy', advimageCalc_data)


def load_images(dataset='mnist', perturb='fgsm'):
    """Load and return original and adversarial images."""

    if dataset == 'mnist':
        dataset_folder = 'MNIST Data'
        data = mnist
    elif dataset == 'fashion':
        dataset_folder = 'Fashion_MNIST Data'
        data = fashion_mnist
    elif dataset == 'cifar10':
        dataset_folder = 'CIFAR10 Data'
        data = cifar10

    # Load original datasets
    _, (images, labels) = data.load_data()

    if dataset == 'mnist' or dataset == 'fashion':
        images = images.reshape(10000, 28, 28, 1)  # For MNIST and Fashion-MNIST

    images = images.astype('float32')
    images = images / 255

    # Load adversarial images
    adversarial_images = np.load(dataset_folder + '/' + dataset + '_' + perturb + '_adv.npy')

    return images, adversarial_images


def calc(images, adversarial_images, measure='sampen'):
    """Calculate and returns the nonlinear measure of both original and adversarial images.
    
    Set measure to what you want to calculate.
    'sampen'  :  Sample entropy
    'frac'    :  Correlation/Fractal dimension
    'hurst'   :  Hurst exponent
    'lyapr'   :  Largest Lyapunov exponent using Rosenstein et al. methods
    
    Docs      :  https://cschoel.github.io/nolds/
    
    If the adversarial image is found to be NaN, we output 0.
    The reason some adversarial iamges are NaN is because
    adversarial generation were unsuccessful for them.
    There is a maximum iteration one can set for adversarial
    generation, the program outputs NaN when the max iteration
    is reached before an adversarial perturbation is found.
    
    For more info look at "adversarial_gen.ipynb"
    """

    imageCalc_data = []
    advimageCalc_data = []

    for i in tqdm(range(len(images))):
        image = images[i]
        image = image.flatten()
        advimage = adversarial_images[i]
        advimage = advimage.flatten()

        if measure == 'sampen':
            imageCalc_data.append(nolds.sampen(image))
            if np.isnan(np.sum(advimage)):
                advimageCalc_data.append(0)
            else:
                advimageCalc_data.append(nolds.samepn(advimage))

        elif measure == 'frac':
            imageCalc_data.append(nolds.corr_dim(image, 1))
            if np.isnan(np.sum(advimage)):
                advimageCalc_data.append(0)
            else:
                advimageCalc_data.append(nolds.corr_dim(advimage, 1))

        elif measure == 'hurst':
            imageCalc_data.append(nolds.hurst_rs(image))
            if np.isnan(np.sum(advimage)):
                advimageCalc_data.append(0)
            else:
                advimageCalc_data.append(nolds.hurst_rs(advimage))

        elif measure == 'lyapr':
            imageCalc_data.append(nolds.lyap_r(image))
            if np.isnan(np.sum(advimage)):
                advimageCalc_data.append(0)
            else:
                advimageCalc_data.append(nolds.lyap_r(advimage))

    return imageCalc_data, advimageCalc_data


if __name__ == "__main__":
    main()
