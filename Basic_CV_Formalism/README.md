# Tutorial on the Cutler-Valisneri Formalism

## Preliminaries

None. This notebook is self contained

## Purpose

This python notebook should be used as a tutorial if one is interested in systematic studies for gravitational wave astronomy using the Cutler-Valisneri (CV) algorithm. This notebook uses the CV algorithm to compute both the bias on parameters from waveform errors and the fluctuation from the true parameters caused by the noise realisation. 

## What is inside..?

Here we use a TaylorF2 waveform and perform parameter estimation using a Metropolis Hasting algorithm. A Fisher matrix is calculated and compared to the result coming from MCMC. Then the CV algorithm is used to predict the bias on the recovered parameters. This notebook is built purely as a tutorial notebook. 
