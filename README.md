# gj-flap-data

The goal of this notebook is processing the CMU pronouncing dictionary to create (input, output) / (underlying, surface) pairs for learning the American English flapping rule discussed in Gildea & Jurafsky's 1996 paper "Learning bias and phonological-rule induction".

## Dependencies

 - **Transcriptions:** The version of the CMU pronouncing dictionary processed here (and assumed to be in the working directory) is taken from https://github.com/emeinhardt/cmu-ipa. Please see the documentation there for more on what processing goes into that file.
 - **`Unix`-like OS:** Some Unix-like shell commands are used throughout, though they aren't essential.
 
 ## Outputs
 
 Currently the only output of the notebook is a `.tsv` file with positive examples of flappable underyling representationss and the flapped surface representations.
