# SPDX-License-Identifier: Apache-2.0
"""Sign-flip test of per-pulse values, at a fixed sample size or with repeated looks.

Each pulse gives one value per channel (for the online mode, the projection of the response
window on the channel template minus that of the baseline segment before the pulse). Under
the null hypothesis (no response) every value is as likely with either sign. The statistic
is the one-sample ``t`` of the values (one-sided: a response in the direction of the
template makes it positive); its null distribution comes from random sign flips of whole
pulses (spec §5.6).

Sequential version (spec §5.13, online mode): the statistic is looked at after every pulse,
or whenever the caller asks. The family-wise error over all looks is held at ``alpha`` by an
alpha-spending function (Lan and DeMets 1983, in its O'Brien-Fleming form) applied to the
permutation trajectories themselves: the observed values and the ``n_permutations``
sign-flipped copies form ``n_permutations + 1`` exchangeable trajectories; at a look with
information fraction ``t`` (pulses so far / ``max_n``) at most
``floor(alpha(t) (n_permutations + 1))`` of them may have crossed so far, and those that
cross are the largest statistics among the trajectories that have not crossed yet (ties at
the boundary do not cross). The observed trajectory is declared significant when it
crosses. Whatever the number and timing of the looks, the
probability that it crosses under the null hypothesis is at most
``floor(alpha (n_permutations + 1)) / (n_permutations + 1) <= alpha``. With a single look at
full information it gives the decision of :func:`sign_flip_test` with the same
``random_state``.
"""

import numpy as np
from scipy.stats import norm

from evoparam._typing import Any, FloatArray, NDArray, RandomState
from evoparam.errors import DataError, SettingsError

__all__ = ["sign_flip_test", "SequentialSignFlip", "OnlineExistence", "alpha_spending"]


def alpha_spending(t: Any, alpha: float = 0.05) -> NDArray[Any]:
    """Cumulative error spent at information fraction ``t`` (0 to 1).

    The Lan-DeMets form of the O'Brien-Fleming boundary, ``2 - 2 Phi(z_{1 - alpha/2} / sqrt(t))``
    (almost nothing early, ``alpha`` at ``t = 1``).
    """
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        out = 2.0 - 2.0 * norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(t))
    return np.where(t >= 1.0, alpha, np.minimum(out, alpha))


def _check(alpha: Any, n_permutations: Any) -> None:
    if not 0 < alpha < 1:
        raise SettingsError("alpha must be in (0, 1).")
    if int(n_permutations) < 1:
        raise SettingsError("n_permutations must be at least 1.")


def _signs(rng: np.random.Generator, n_permutations: Any) -> Any:
    """Signs of one pulse in each permutation (the same draws online and offline)."""
    return 2.0 * rng.integers(0, 2, size=n_permutations) - 1.0


def _statistic(sums: Any, sum_sq: Any, n: Any) -> NDArray[Any]:
    """One-sample t of each trajectory (rows) and channel (columns); -inf when undefined."""
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / n
        var = (sum_sq - sums**2 / n) / (n - 1)
        t = mean / np.sqrt(np.clip(var, 0.0, None) / n)
    return np.where(np.isnan(t) | (n < 2), -np.inf, t)


def _allowed(alpha: Any, size: Any) -> Any:
    return np.floor(np.asarray(alpha) * size + 1e-9).astype(int)


def sign_flip_test(
    values: Any,
    *,
    alpha: float = 0.05,
    n_permutations: int = 1000,
    random_state: RandomState | None = None,
) -> dict[str, Any]:
    """Sign-flip permutation test of per-pulse values, at a fixed sample size.

    Parameters
    ----------
    values : array [n_pulses] or [n_pulses, n_channels]
        One value per pulse (and channel); NaN values are left out.
    alpha : float
    n_permutations : int
    random_state : int or Generator, optional

    Returns
    -------
    dict of arrays [n_channels]
        ``statistic`` (one-sample t), ``p_value`` (``(1 + #{null >= observed}) /
        (1 + n_permutations)``), ``significant`` (``p_value <= alpha``) and ``n`` (values
        used).
    """
    _check(alpha, n_permutations)
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise DataError("values must be [n_pulses] or [n_pulses, n_channels].")
    rng = np.random.default_rng(random_state)
    n_perm = int(n_permutations)
    sums = np.zeros((n_perm + 1, values.shape[1]))
    present = np.isfinite(values)
    v = np.where(present, values, 0.0)
    for row in v:
        signs = _signs(rng, n_perm)
        sums[0] += row
        sums[1:] += signs[:, None] * row[None]
    n = present.sum(axis=0)
    stat = _statistic(sums, (v**2).sum(axis=0), n)
    count = (stat[1:] >= stat[0] - 1e-12).sum(axis=0)
    p = (1.0 + count) / (1.0 + n_perm)
    p = np.where(n >= 2, p, np.nan)
    return {
        "statistic": stat[0],
        "p_value": p,
        "significant": np.nan_to_num(p, nan=1.0) <= alpha,
        "n": n,
    }


class SequentialSignFlip:
    """Running sign-flip test with an alpha-spending rule over repeated looks.

    Parameters
    ----------
    n_channels : int
    alpha : float
        Family-wise error over all looks, per channel.
    max_n : int
        Planned number of pulses: the information fraction of a look is ``n / max_n``
        (capped at 1; after ``max_n`` pulses no more error is spent).
    n_permutations : int
    random_state : int or Generator, optional

    Notes
    -----
    :meth:`update` adds one pulse (one value per channel; NaN: no value for that channel);
    :meth:`look` applies the boundary and returns the decisions. ``look(final=True)``
    declares the end of the stream and spends what remains of ``alpha``. A channel that has
    crossed stays significant. The cost of a pulse is ``O(n_permutations x n_channels)``.
    """

    def __init__(
        self,
        n_channels: int,
        *,
        alpha: float = 0.05,
        max_n: int = 100,
        n_permutations: int = 1000,
        random_state: Any = None,
    ) -> None:
        _check(alpha, n_permutations)
        if int(max_n) < 2:
            raise SettingsError("max_n must be at least 2.")
        self.n_channels: int = int(n_channels)
        self.alpha: float = float(alpha)
        self.max_n: int = int(max_n)
        self.n_permutations: int = int(n_permutations)
        self.random_state = random_state
        self.reset()

    def reset(self) -> None:
        """Forget every pulse and look."""
        n_traj = self.n_permutations + 1
        self._rng = np.random.default_rng(self.random_state)
        self._sums = np.zeros((n_traj, self.n_channels))
        self._sum_sq = np.zeros(self.n_channels)
        self.n: Any = np.zeros(self.n_channels, int)
        self._alive = np.ones((n_traj, self.n_channels), bool)
        self._crossed = np.zeros(self.n_channels, int)
        self.significant: NDArray[Any] = np.zeros(self.n_channels, bool)
        self.alpha_spent: FloatArray | NDArray[Any] = np.zeros(self.n_channels)
        self.n_looks: int = 0
        self._values: list[Any]
        self._values = []

    @property
    def values(self) -> Any:
        """The values received so far, [n_pulses, n_channels]."""
        return np.array(self._values).reshape(-1, self.n_channels)

    def update(self, values: Any) -> None:
        """Add one pulse: one value per channel (NaN: none for that channel)."""
        values = np.asarray(values, float).ravel()
        if values.size != self.n_channels:
            raise DataError(f"update expects {self.n_channels} values, got {values.size}.")
        present = np.isfinite(values)
        v = np.where(present, values, 0.0)
        signs = _signs(self._rng, self.n_permutations)
        self._sums[0] += v
        self._sums[1:] += signs[:, None] * v[None]
        self._sum_sq += v**2
        self.n += present
        self._values.append(values)

    def statistics(self) -> tuple[NDArray[Any], NDArray[Any]]:
        """Statistic of the observed trajectory and fixed-sample p-value, per channel."""
        stat = _statistic(self._sums, self._sum_sq, self.n)
        count = (stat[1:] >= stat[0] - 1e-12).sum(axis=0)
        p = np.where(self.n >= 2, (1.0 + count) / (1.0 + self.n_permutations), np.nan)
        return stat, p

    def look(self, final: bool = False) -> dict[str, Any]:
        """Apply the boundary at the current information fraction.

        Returns
        -------
        dict of arrays [n_channels]
            ``significant`` (crossed at this or an earlier look), ``statistic``, ``p_value``
            (fixed-sample permutation p-value, for information; not adjusted for the
            looks), ``alpha_spent`` and ``n``.
        """
        stat, p = self.statistics()
        n_traj = self.n_permutations + 1
        fraction = np.ones(self.n_channels) if final else np.minimum(self.n / self.max_n, 1.0)
        spent = np.where(self.n >= 2, alpha_spending(fraction, self.alpha), 0.0)
        spent = np.maximum(spent, self.alpha_spent)
        room = _allowed(spent, n_traj) - self._crossed
        cols = np.flatnonzero(room > 0)
        if len(cols):
            # A trajectory crosses when fewer than room + 1 live ones are at least as large,
            # i.e. when it exceeds the (room + 1)-th largest live statistic (crossed
            # trajectories count as -inf, below every live one)
            live = np.where(self._alive[:, cols], stat[:, cols], -np.inf)
            ordered = np.sort(live, axis=0)
            rank = np.clip(n_traj - 1 - room[cols], 0, n_traj - 1)
            bound = np.where(room[cols] >= n_traj, -np.inf, ordered[rank, np.arange(len(cols))])
            cross = (live > bound[None]) & np.isfinite(live)
            self._alive[:, cols] &= ~cross
            self._crossed[cols] += cross.sum(axis=0)
            self.significant[cols] |= cross[0]
        self.alpha_spent = spent
        self.n_looks += 1
        return {
            "significant": self.significant.copy(),
            "statistic": stat[0],
            "p_value": p,
            "alpha_spent": spent.copy(),
            "n": self.n.copy(),
        }


class OnlineExistence:
    """Per-pulse and running existence statistics of the online mode (spec §10.6).

    Each pulse brings, per channel, the projection of its response window on the unit-norm
    channel template (``projection``) and that of the baseline segment before it
    (``baseline_projection``).

    - Per pulse: ``z = (projection - mean of the baseline projections so far) / their SD``
      (NaN until two baselines have been seen), and ``present = z > threshold_sd``.
    - Running: :class:`SequentialSignFlip` on ``projection - baseline_projection`` (a pulse
      without baseline is left out), looked at after every pulse when ``looks="pulse"``.

    Parameters are those of :class:`SequentialSignFlip`, plus ``threshold_sd`` and ``looks``
    (``"pulse"`` or None: only when :meth:`look` is called).
    """

    def __init__(
        self,
        n_channels: int,
        *,
        threshold_sd: float = 3.0,
        alpha: float = 0.05,
        max_n: int = 100,
        n_permutations: int = 1000,
        looks: str = "pulse",
        random_state: Any = None,
    ) -> None:
        if looks not in ("pulse", None):
            raise SettingsError("looks must be 'pulse' or None.")
        self.threshold_sd: float = float(threshold_sd)
        self.looks = looks
        self.test: SequentialSignFlip = SequentialSignFlip(
            n_channels,
            alpha=alpha,
            max_n=max_n,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        self.reset()

    def reset(self) -> None:
        """Reset the running test to no trial."""
        n_ch = self.test.n_channels
        self.test.reset()
        self._count = np.zeros(n_ch)
        self._mean = np.zeros(n_ch)
        self._m2 = np.zeros(n_ch)
        self.last: dict[str, Any] | None = None

    def update(
        self, projection: Any, baseline_projection: Any
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """Add one pulse; returns ``(z, present)``, arrays [n_channels]."""
        projection = np.asarray(projection, float)
        base = np.asarray(baseline_projection, float)
        seen = np.isfinite(base)
        # Welford update of the baseline mean and variance
        self._count += seen
        delta = np.where(seen, base - self._mean, 0.0)
        self._mean += delta / np.maximum(self._count, 1)
        self._m2 += delta * np.where(seen, base - self._mean, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sd = np.sqrt(self._m2 / (self._count - 1))
            z = np.where((self._count >= 2) & (sd > 0), (projection - self._mean) / sd, np.nan)
        self.test.update(projection - base)
        if self.looks == "pulse":
            self.last = self.test.look()
        return z, np.nan_to_num(z, nan=-np.inf) > self.threshold_sd

    def look(self, final: bool = False) -> dict[str, Any]:
        """Return a look of the running test (see :meth:`SequentialSignFlip.look`)."""
        last = self.last = self.test.look(final=final)
        return last
