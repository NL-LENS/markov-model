# Life event sequence Markov model

This repository contains a model for generating life event sequences with a Markov model. Because the model uses explicit probability distributions to generate life event sequences, it can be used to compute the Monte Carlo irreducible error. In this way, the generated sequences can be used to validate the predictions of another model (e.g., a transformer-based sequence model).

Requires numpy.

## Usage

The `generate_sequences.py` script generates life event sequences that are stored in a CSV file based on Markov model configurations in a YAML file (the default is `config.yaml`). For example:

```bash
python generate_sequences.py --n 1000 --seed 0 --out sequences.csv
```

To calculate the irreducible (Bayes) error, add the `--bayes-error` flat.

## Model

The Markov model can combine different mechanisms to generate life event sequences. By default, the event sequences start with an initial distribution and follow a transition matrix. Optionally, cumulative time stamps that follow an exponential distribution can be added to the sequence that change the event transition probabilities. Another option is to add event context variables (attributes). Context variables can be continuous, categorical, or ordered categorical.

- Ordered categorical attributes increase every time the corresponding event occurs (e.g., education level increases every time a new education spell starts)
- Categorical attributes have categories with probabilities conditional on the previous event
- Continuous attributes are sampled from a normal distribution with means conditional on a categorical attribute associated with the same event (e.g., salary depends on current company)

Moreover, the model tracks the state of specific event spells. For example, an education spell must end before the next can start.

In sum, the sequences can be used to validate that a model captures multiple mechanisms:

- Event transition probabilities
- Effect of time on event transition probabilities
- Conditional event attribute probabilities / means
- Causal order of events
