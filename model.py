"""Markovian model for simulating simplified life-event sequences.

The simulator generates event sequences from a first-order Markov process over
an *augmented* state: besides the current event it tracks the currently-active
education degree and job/company.
Events come in start/end pairs (``start_education``/``end_education`` and
``start_job``/``end_job``); you cannot start a second education/job before
ending the active one, so the base transition probabilities are **masked** by
the active state at each step and renormalized.

Some attributes are therefore deterministic functions of history:

* ``start_education``'s ``degree`` advances in a fixed order by the number of
  completed educations (primary -> secondary -> vocational -> university);
* ``start_job``'s ``company`` is sampled (its distribution may be conditioned on
  the previous event), and ``salary`` is drawn conditional on that company.

End events (``end_education``/``end_job``) carry no attributes; they only clear
the active resource so the next start becomes legal.

When timestamps are enabled, transitions are additionally modulated by absolute
age ``t`` (cumulative time since the sequence start): the legal transition row is
tilted by ``exp(beta * t)`` and renormalized, with per-edge slopes ``beta`` from
the config's ``time_betas`` (default 0, i.e. time-independent). Time therefore
becomes informative about the next event, and the Bayes-optimal next-event
predictor conditions on ``t``.

Because every distribution is specified explicitly, the Bayes error of
predicting the next event and each attribute can be estimated by Monte-Carlo:
simulate, then score the known optimal predictor against the realised values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Simulation state (the "active events")
# ---------------------------------------------------------------------------
@dataclass
class SimState:
    """Mutable per-sequence state threaded through one simulation.

    ``active`` maps a resource name (e.g. ``"education"``) to ``True`` while the
    resource is active and ``None`` when it is inactive (only the flag matters,
    for transition masking). ``completed`` counts how many times each resource
    has been ended (used for ordered attributes). ``scratch`` holds values
    sampled earlier in the *current* step so sibling attributes can depend on
    them (e.g. ``salary`` reading the just-sampled ``company``).
    """

    active: dict[str, bool | None]
    completed: dict[str, int]
    scratch: dict[str, str | float] = field(default_factory=dict)
    prev_event: str | None = None


# ---------------------------------------------------------------------------
# Attribute specifications
# ---------------------------------------------------------------------------
class AttributeSpec(Protocol):
    """An attribute attached to an event.

    Concrete specs know how to *sample* a value given the simulation state and
    how to report the Bayes-optimal *prediction* of that value (used for
    Monte-Carlo Bayes-error estimation).
    """

    name: str

    def sample(self, rng: np.random.Generator, state: SimState) -> str | float:
        ...

    def optimal_prediction(self, state: SimState) -> str | float:
        ...


@dataclass(frozen=True)
class OrderedCategoricalSpec:
    """Deterministic categorical that advances in order by completed count.

    The value of the n-th started resource is ``values[n]`` where ``n`` is the
    number of times the resource has already been completed. Fully determined by
    history, so its Bayes error is zero.
    """

    name: str
    values: list[str]
    resource: str

    def sample(self, rng: np.random.Generator, state: SimState) -> str:
        return self.values[state.completed[self.resource]]

    def optimal_prediction(self, state: SimState) -> str:
        return self.values[state.completed[self.resource]]


@dataclass(frozen=True)
class CategoricalSpec:
    """A sampled categorical attribute (e.g. the company of a new job).

    Optionally conditioned on the *previous* event: ``probs_by_prev`` maps a
    previous-event name to its own distribution, and ``probs`` is the fallback
    used for previous events not listed (and for the first event, where there is
    no predecessor). When ``probs_by_prev`` is empty the attribute is
    unconditional. The Bayes-optimal prediction is the mode of whichever
    distribution applies, so the estimated error is the prev-event-averaged
    ``1 - max(p)``.
    """

    name: str
    values: list[str]
    probs: np.ndarray
    probs_by_prev: dict[str | None, np.ndarray] = field(default_factory=dict)

    def _probs(self, state: SimState) -> np.ndarray:
        return self.probs_by_prev.get(state.prev_event, self.probs)

    def sample(self, rng: np.random.Generator, state: SimState) -> str:
        return str(rng.choice(self.values, p=self._probs(state)))

    def optimal_prediction(self, state: SimState) -> str:
        return self.values[int(np.argmax(self._probs(state)))]


@dataclass(frozen=True)
class ConditionalNormalSpec:
    """A continuous attribute conditioned on a sibling categorical attribute.

    ``salary ~ Normal(means[company], sd)``. The Bayes-optimal point prediction
    is the conditional mean, leaving the irreducible MSE ``sd**2``.
    """

    name: str
    condition_on: str
    means: dict[str, float]
    sd: float

    def sample(self, rng: np.random.Generator, state: SimState) -> float:
        key = state.scratch[self.condition_on]
        return float(rng.normal(loc=self.means[key], scale=self.sd))

    def optimal_prediction(self, state: SimState) -> float:
        key = state.scratch[self.condition_on]
        return float(self.means[key])


@dataclass(frozen=True)
class Resource:
    """A start/end event pair that shares a single active value."""

    name: str
    start_event: str
    end_event: str
    attribute: str
    n_values: int  # capacity for ordered resources (max completions)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MarkovEventModel:
    def __init__(
        self,
        event_names: list[str],
        initial: np.ndarray,
        transition: np.ndarray,
        attributes: dict[str, list[AttributeSpec]],
        resources: list[Resource],
        time_rate: float = 1.0,
        time_beta: np.ndarray | None = None,
        generate_attributes: bool = True,
        generate_timestamps: bool = True,
    ):
        self.event_names = event_names
        self.num_events = len(event_names)
        self.idx_of = {name: i for i, name in enumerate(event_names)}

        self.initial = np.asarray(initial, dtype=float)
        self.transition = np.asarray(transition, dtype=float)
        # Per-edge age slopes; zeros => time-independent transitions.
        self.time_beta = (
            np.zeros((self.num_events, self.num_events), dtype=float)
            if time_beta is None
            else np.asarray(time_beta, dtype=float)
        )
        self.attributes = attributes
        self.resources = resources

        self.time_rate = time_rate
        self.generate_attributes = generate_attributes
        self.generate_timestamps = generate_timestamps

        self.death_idx = self.idx_of["death"]
        # Resources indexed by their start/end event for quick lookup.
        self._resource_by_start = {r.start_event: r for r in resources}
        self._resource_by_end = {r.end_event: r for r in resources}

    # ------------------------------------------------------------------
    # Construction from YAML
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "MarkovEventModel":
        with open(path) as fh:
            config = yaml.safe_load(fh)
        return cls.from_config(config, **overrides)

    @classmethod
    def from_config(cls, config: dict, **overrides) -> "MarkovEventModel":
        events = config["events"]
        event_names = list(events.keys())
        idx_of = {name: i for i, name in enumerate(event_names)}
        n = len(event_names)

        # Initial distribution.
        initial = np.array([events[name].get("initial", 0.0) for name in event_names], dtype=float)
        if initial.sum() <= 0:
            raise ValueError("At least one event must have a positive `initial` probability.")
        initial /= initial.sum()

        # Base transition matrix.
        transition = np.zeros((n, n), dtype=float)
        for name, spec in events.items():
            i = idx_of[name]
            if spec.get("absorbing"):
                transition[i, i] = 1.0
                continue
            for target, prob in (spec.get("transitions") or {}).items():
                transition[i, idx_of[target]] = prob
            row_sum = transition[i].sum()
            if row_sum <= 0:
                raise ValueError(f"Event {name!r} has no outgoing transitions.")
            transition[i] /= row_sum

        # Per-edge age slopes for the timestamp-dependent transition tilt.
        time_beta = np.zeros((n, n), dtype=float)
        for name, spec in events.items():
            for target, beta in (spec.get("time_betas") or {}).items():
                time_beta[idx_of[name], idx_of[target]] = beta

        # Resources.
        attr_defs = config.get("attributes", {})
        resources = [
            Resource(
                name=rname,
                start_event=rspec["start_event"],
                end_event=rspec["end_event"],
                attribute=rspec["attribute"],
                n_values=len(attr_defs[rspec["attribute"]].get("values", [])),
            )
            for rname, rspec in config.get("resources", {}).items()
        ]
        resource_of_attr = {r.attribute: r for r in resources}

        # Attribute specs, built per event so start/end events of a resource get
        # the right behavior (sample-and-activate vs copy-active).
        attributes: dict[str, list[AttributeSpec]] = {}
        for name, spec in events.items():
            specs: list[AttributeSpec] = []
            for attr_name in spec.get("attributes") or []:
                specs.append(cls._build_attr_spec(attr_name, attr_defs, resources))
            attributes[name] = specs

        model = cls(
            event_names=event_names,
            initial=initial,
            transition=transition,
            attributes=attributes,
            resources=resources,
            time_rate=config.get("time_rate", 1.0),
            time_beta=time_beta,
            **overrides,
        )
        return model

    @staticmethod
    def _build_attr_spec(
        attr_name: str,
        attr_defs: dict,
        resources: list[Resource],
    ) -> AttributeSpec:
        definition = attr_defs[attr_name]
        kind = definition["type"]

        resource_by_attr = {r.attribute: r for r in resources}

        if kind == "ordered":
            return OrderedCategoricalSpec(
                name=attr_name,
                values=list(definition["values"]),
                resource=resource_by_attr[attr_name].name,
            )
        if kind == "categorical":
            return CategoricalSpec(
                name=attr_name,
                values=list(definition["values"]),
                probs=_normalize(definition["probs"]),
                probs_by_prev={
                    prev: _normalize(vec)
                    for prev, vec in (definition.get("probs_by_prev") or {}).items()
                },
            )
        if kind == "normal":
            return ConditionalNormalSpec(
                name=attr_name,
                condition_on=definition["condition_on"],
                means={k: float(v) for k, v in definition["means"].items()},
                sd=float(definition["sd"]),
            )
        raise ValueError(f"Unknown attribute type {kind!r} for {attr_name!r}.")

    # ------------------------------------------------------------------
    # Active-state masking
    # ------------------------------------------------------------------
    def _legal_mask(self, state: SimState) -> np.ndarray:
        """Boolean mask of which next events are legal given the active state."""
        mask = np.ones(self.num_events, dtype=bool)
        for r in self.resources:
            active = state.active[r.name] is not None
            # Cannot start a resource that is already active, nor (for ordered
            # resources) start one once every value has been used.
            exhausted = state.completed[r.name] >= r.n_values if r.n_values else False
            if active or exhausted:
                mask[self.idx_of[r.start_event]] = False
            # Cannot end a resource that is not active.
            if not active:
                mask[self.idx_of[r.end_event]] = False
        return mask

    def _masked_transition(
        self, idx: int, state: SimState, time: float | None = None
    ) -> np.ndarray:
        """Transition row from ``idx`` with illegal targets removed + renormalized.

        When ``time`` is given, the legal row is tilted by ``exp(beta * time)``
        (age-dependent transitions); ``time=None`` leaves it time-independent.
        """
        row = self.transition[idx] * self._legal_mask(state)
        if time is not None:
            logits = self.time_beta[idx] * time
            # Subtract the max over legal edges so exp() can't overflow; masked
            # edges keep base 0 and stay 0 after the multiply.
            logits = logits - logits[row > 0].max()
            row = row * np.exp(logits)
        total = row.sum()
        if total <= 0:
            # Everything was masked out: fall back to death (always reachable).
            row = np.zeros(self.num_events)
            row[self.death_idx] = 1.0
            return row
        return row / total

    def _fresh_state(self) -> SimState:
        return SimState(
            active={r.name: None for r in self.resources},
            completed={r.name: 0 for r in self.resources},
        )

    def _apply_event(
        self, event: str, state: SimState, rng: np.random.Generator
    ) -> list[tuple[str, str | float]]:
        """Sample/emit attributes for ``event`` and update the active state.

        Returns the list of (attribute_name, value) pairs. Active-state updates
        (set/clear/complete) happen here so masking on the *next* step is
        correct, regardless of whether attribute values are emitted.
        """
        state.scratch.clear()

        # Activate a resource as it starts.
        start_resource = self._resource_by_start.get(event)
        attrs: list[tuple[str, str | float]] = []

        for spec in self.attributes[event]:
            value = spec.sample(rng, state)
            state.scratch[spec.name] = value
            attrs.append((spec.name, value))

        # Update active state based on the resource role of this event. Only the
        # active/inactive flag matters (for transition masking); the value itself
        # is not carried onto the end event anymore.
        if start_resource is not None:
            state.active[start_resource.name] = True
        end_resource = self._resource_by_end.get(event)
        if end_resource is not None:
            state.active[end_resource.name] = None
            state.completed[end_resource.name] += 1

        return attrs if self.generate_attributes else []

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _simulate(self, rng: np.random.Generator, max_len: int):
        """Yield ``(time, event, attrs, masked_dist)`` for one sequence.

        ``masked_dist`` is the legal next-event distribution actually used to
        sample the transition out of this step (``None`` on the terminal step).
        Shared by the public formatter and the Monte-Carlo scorers.
        """
        state = self._fresh_state()
        idx = int(rng.choice(self.num_events, p=self.initial))
        time: float | None = rng.exponential(self.time_rate) if self.generate_timestamps else None

        for _ in range(max_len):
            event = self.event_names[idx]
            attrs = self._apply_event(event, state, rng)

            if self.transition[idx, idx] == 1.0:  # absorbing (death)
                yield time, event, attrs, None
                return

            dist = self._masked_transition(idx, state, time)
            yield time, event, attrs, dist

            state.prev_event = event
            idx = int(rng.choice(self.num_events, p=dist))
            if self.generate_timestamps:
                time = time + rng.exponential(self.time_rate)

    def simulate_sequence(
        self, rng: np.random.Generator, max_len: int = 50
    ) -> list[tuple[float | None, str, list[tuple[str, str | float]]]]:
        """Simulate one sequence.

        Returns a list of ``(time, event, attributes)`` tuples, where ``time`` is
        the cumulative timestamp (or ``None`` when timestamps are disabled) and
        ``attributes`` is a list of ``(name, value)`` pairs (empty when
        attributes are disabled or the event carries none).
        """
        return [(t, ev, attrs) for t, ev, attrs, _ in self._simulate(rng, max_len)]

    # ------------------------------------------------------------------
    # Monte-Carlo Bayes error
    # ------------------------------------------------------------------
    def bayes_error_events(
        self, rng: np.random.Generator, n_sequences: int = 10_000, max_len: int = 50
    ) -> float:
        """Estimate the irreducible next-event misclassification rate.

        At each step the Bayes-optimal predictor picks ``argmax`` of the legal
        (masked) next-event distribution; we compare it to the realised next
        event and return the error rate.
        """
        errors = 0
        total = 0
        for _ in range(n_sequences):
            steps = list(self._simulate(rng, max_len))
            for (_, _, _, dist), (_, nxt_event, _, _) in zip(steps[:-1], steps[1:]):
                predicted = int(np.argmax(dist))
                errors += int(predicted != self.idx_of[nxt_event])
                total += 1
        return errors / total if total else float("nan")

    def bayes_error_attributes(
        self, rng: np.random.Generator, n_sequences: int = 10_000, max_len: int = 50
    ) -> dict[str, float]:
        """Estimate the irreducible error of predicting each attribute.

        Keyed by ``"<event>.<attribute>"`` since an attribute name can play
        different roles on different events. 0/1 error for categorical/
        deterministic attributes (so the deterministic ``degree`` comes out ~0),
        squared error for the continuous ``salary`` (Bayes MSE ``= sd**2``). The
        optimal prediction is evaluated against the same realised state used to
        sample the value. End events carry no attributes and so do not appear.
        """
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        for _ in range(n_sequences):
            state = self._fresh_state()
            idx = int(rng.choice(self.num_events, p=self.initial))
            # Track age so the sampled path matches the real (age-tilted) process.
            time = rng.exponential(self.time_rate) if self.generate_timestamps else None
            for _ in range(max_len):
                event = self.event_names[idx]
                state.scratch.clear()
                for spec in self.attributes[event]:
                    key = f"{event}.{spec.name}"
                    prediction = spec.optimal_prediction(state)
                    value = spec.sample(rng, state)
                    state.scratch[spec.name] = value
                    loss = _loss(spec, prediction, value)
                    sums[key] = sums.get(key, 0.0) + loss
                    counts[key] = counts.get(key, 0) + 1

                start_resource = self._resource_by_start.get(event)
                if start_resource is not None:
                    state.active[start_resource.name] = True
                end_resource = self._resource_by_end.get(event)
                if end_resource is not None:
                    state.active[end_resource.name] = None
                    state.completed[end_resource.name] += 1

                state.prev_event = event
                if self.transition[idx, idx] == 1.0:
                    break
                idx = int(rng.choice(self.num_events, p=self._masked_transition(idx, state, time)))
                if time is not None:
                    time = time + rng.exponential(self.time_rate)

        return {name: sums[name] / counts[name] for name in sums}


def _normalize(vec) -> np.ndarray:
    """Coerce a probability vector to a float array summing to 1."""
    arr = np.asarray(vec, dtype=float)
    total = arr.sum()
    if total <= 0:
        raise ValueError("Probability vector must have a positive sum.")
    return arr / total


def _loss(spec: AttributeSpec, prediction: str | float, value: str | float) -> float:
    """0/1 loss for categorical specs, squared error for continuous specs."""
    if isinstance(spec, ConditionalNormalSpec):
        return float((prediction - value) ** 2)
    return float(prediction != value)
