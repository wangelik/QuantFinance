# import table
from dataclasses import dataclass

import numpy as np


@dataclass
class StockSimulation:
    """General class to simulate stock returns using different distributions

    Properties:
        mu      (float): estimated annualized return
        sigma   (float): estimated standard deviation of annualized returns
        start   (float): start price
        reps    (int): simulation size / repetition
        dt      (float): time step size (parts of year)
        n       (int): number of steps simulated (per dt)
        verbose (bool): print/do not print information
        rets    (np.array(reps, n)): simulated raw returns (per dt)
        logrets (np.array(reps, n)): simulated log returns (per dt)
        prices  (np.array(reps, n)): simulated price paths (per dt)

    Methods:
        gaussian    assume normal (Gaussian) raw returns
        gbm         assume log-normal raw returns (Geometric Brownian Motion)

    Note:
        Class remembers used algorithms, internal states and sends alerts
        if settings are changed during runtime.
    """

    # init settings
    _mu: float
    _sigma: float
    _start: float = 100
    _reps: int = 100
    _dt: float = 1 / 252
    _n: int = 252
    _algo: str = "Empty"
    _changed: bool = False
    verbose: bool = True
    # init data containers
    _rets: np.ndarray = np.array([])
    _logrets: np.ndarray = np.array([])
    _prices: np.ndarray = np.array([])

    # define setting properties
    @property
    def mu(self) -> float:
        return self._mu

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def start(self) -> float:
        return self._start

    @property
    def reps(self) -> int:
        return self._reps

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def n(self) -> int:
        return self._n

    # define result properties
    @property
    def rets(self) -> np.ndarray:
        if self._changed:
            print(
                "[Note] Settings had been changed and are not in sync with results. You may want to rerun simulation."
            )
        return self._rets

    @property
    def logrets(self) -> np.ndarray:
        if self._changed:
            print(
                "[Note] Settings had been changed and are not in sync with results. You may want to rerun simulation."
            )
        return self._logrets

    @property
    def prices(self) -> np.ndarray:
        if self._changed:
            print(
                "[Note] Settings had been changed and are not in sync with results. You may want to rerun simulation."
            )
        return self._prices

    # define setting setters
    @mu.setter
    def mu(self, val: float) -> None:
        self._changed = True
        self._mu = val

    @sigma.setter
    def sigma(self, val: float) -> None:
        self._changed = True
        self._sigma = val

    @start.setter
    def start(self, val: float) -> None:
        self._changed = True
        self._start = val

    @reps.setter
    def reps(self, val: int) -> None:
        self._changed = True
        self._reps = val

    @dt.setter
    def dt(self, val: float) -> None:
        self._changed = True
        self._dt = val

    @n.setter
    def n(self, val: int) -> None:
        self._changed = True
        self._n = val

    def gaussian(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Timestep-based Gaussian stock return simulation

        Returns:
            rets    (np.array(reps, n)): simulated raw returns (per dt)
            logrets (np.array(reps, n)): simulated log returns (per dt)
            prices  (np.array(reps, n)): simulated price paths (per dt)

        Note:
            method motivated by discrete version of GBM PDE:
            delta_S/S = mu * delta_t + sigma * sqrt(delta_t) * N(0,1)
        """

        # printing settings
        self.__vprint(
            f"[+] Running Gaussian Simulation for mu={self._mu}, "
            f"sigma={self._sigma}, dt={self._dt}, n={self._n}, "
            f"reps={self.reps}, start={self._start}"
        )

        # generate gaussian random returns
        self._rets = np.random.normal(
            self._mu * self._dt,
            self._sigma * np.sqrt(self._dt),
            size=(self._reps, self._n),
        )
        # compute log returns
        self._logrets = np.log(self._rets + 1)
        # compute prices (faster than exp + cumsum)
        self._prices = self._start * (self._rets + 1).cumprod(axis=1)

        # update state
        self._changed = False
        self._algo = "Gaussian"

        # return data
        return self._rets, self._logrets, self._prices

    def gbm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Timestep-based Geometric Brownian Motion stock return simulation

        Returns:
            rets    (np.array(reps, n)): simulated raw returns (per dt)
            logrets (np.array(reps, n)): simulated log returns (per dt)
            prices  (np.array(reps, n)): simulated price paths (per dt)

        Note:
            method motivated by exact (continuous) version of GBM PDE (Ito):
            d_S/S = mu * d_t + sigma * d_W(t) =>
            d_S/S = mu * d_t + sigma * sqrt(d_t) * N(0,1) [W(t)] =>
            ln(S_dt/S_0) = (mu - sigma^2/2)dt + sigma * sqrt(d_t) * N(0,1) [W(t)]
        """

        # printing settings
        self.__vprint(
            f"[+] Running GBM Simulation for mu={self._mu}, "
            f"sigma={self._sigma}, dt={self._dt}, n={self._n}, "
            f"reps={self.reps}, start={self._start}"
        )

        # generate GBM random log returns
        self._logrets = np.random.normal(
            (self._mu - (self._sigma**2) / 2) * self._dt,
            self._sigma * np.sqrt(self._dt),
            size=(self._reps, self._n),
        )
        # compute raw returns
        self._rets = np.exp(self._logrets) - 1
        # compute prices (faster than exp + cumsum)
        self._prices = self._start * (self._rets + 1).cumprod(axis=1)

        # update state
        self._changed = False
        self._algo = "GBM"

        # return data
        return self._rets, self._logrets, self._prices

    def __vprint(self, info: str) -> None:
        """Verbose-based print utility function"""
        if self.verbose:
            print(info)

    def __repr__(self) -> str:
        """Customized settings representation"""
        return (
            f"{__class__.__name__} for mu={self._mu}, "
            f"sigma={self._sigma}, dt={self._dt}, n={self._n}, "
            f"reps={self.reps}, start={self._start}, algo={self._algo}"
        )
