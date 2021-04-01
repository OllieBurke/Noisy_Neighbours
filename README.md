# Noisy Neighbours: inference biases from overlapping gravitational-wave signals

Codes to reproduce results in the paper

## Basic\_CV\_Formalism

This contains a simple jupyter notebook intended as a tutorial. This tutorial is for people who are getting started with waveform systematic studies using the Cutler Valisneri formalism.

## ET\_Example

This folder contains a script to predict biases on parameters for an groundbased detector like ET. We use TaylorF2 waveform models and compute biases in the case of inference on two signals with waveform errors, 4 missed signals, and detector noise

## LISA\_Example

Similar to above, we provide a script to predict inference biases on parameters for a space-based detectors. We performm parameter estimation on four signals with waveform errors, a galactic foreground mimicking white dwarf binaries and detector noise.

## Global\_Fit\_Correction

In section 6 of the paper, we propose an algorithm to correct biases on parameter estimates when only a subset of signals are searched over. We outline the algorithm and results in the paper here.

### Disclaimer

These codes were produced independently by the author Ollie Burke. All mistakes that lie within these are his own fault, or, perhaps, his co-authors (Andrea Antonelli)

email: ollie.burke@aei.mpg.de 
