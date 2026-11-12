import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Piecewise cubic IFR interpolator with monotone, convex segments.

    Each interval uses a cubic Hermite polynomial.  For increasing node
    data, its two endpoint slopes are set to ``d - c`` and ``d + c`` where
    ``d`` is the interval secant and ``c`` is bounded by both neighboring
    secant changes and the interval level.  This makes the derivative
    nonnegative and the second derivative nonnegative on that interval.
    If the node data do not support that shape, the interval is kept linear.
    The construction is continuous at every node; derivatives may differ
    across nodes, as in many practical curve interpolators.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        self._x = np.asarray(self.axis1_, dtype=float)
        self._y = np.asarray(self.values_, dtype=float)
        if self.length_ == 0:
            raise ValueError("At least one interpolation node is required")
        if self.length_ > 1 and np.any(np.diff(self._x) <= 0.0):
            raise ValueError("Interpolation nodes must be strictly increasing")
        if self.extrap_method_ not in (ExtrapMethod.FLAT, ExtrapMethod.LINEAR):
            raise ValueError(f"Unsupported extrapolation method: {self.extrap_method_}")

        self._h = np.diff(self._x)
        self._secants = np.diff(self._y) / self._h if self.length_ > 1 else np.array([])
        # Store each interval's endpoint derivatives.  An interval with
        # incompatible node behavior falls back to a straight line.
        self._left_slopes = self._secants.copy()
        self._right_slopes = self._secants.copy()
        for i, d in enumerate(self._secants):
            if d <= 0.0:
                continue
            changes = []
            if i > 0:
                changes.append(d - self._secants[i - 1])
            if i + 1 < len(self._secants):
                changes.append(self._secants[i + 1] - d)
            positive_changes = [change for change in changes if change > 0.0]
            if not positive_changes:
                continue
            # Symmetric slopes yield a nonnegative linear second derivative.
            # The bound also keeps the left slope nonnegative.
            curvature = min(0.25 * min(positive_changes), 0.25 * d)
            self._left_slopes[i] = d - curvature
            self._right_slopes[i] = d + curvature

    def _interval_coefficients(self, i: int):
        """Return coefficients for y(t)=y0+a*t+b*t^2+c*t^3, t in [0, 1]."""
        h = self._h[i]
        y0, y1 = self._y[i], self._y[i + 1]
        m0, m1 = self._left_slopes[i], self._right_slopes[i]
        delta = y1 - y0
        a = h * m0
        b = 3.0 * delta - h * (2.0 * m0 + m1)
        c = -2.0 * delta + h * (m0 + m1)
        return a, b, c

    @staticmethod
    def _primitive(t: float, a: float, b: float, c: float) -> float:
        return a * t**2 / 2.0 + b * t**3 / 3.0 + c * t**4 / 4.0

    def interpolate(self, x: float) -> float:
        x = float(x)
        if self.length_ == 1:
            return float(self._y[0])
        if x < self._x[0]:
            if self.extrap_method_ == ExtrapMethod.FLAT:
                return float(self._y[0])
            return float(self._y[0] + self._secants[0] * (x - self._x[0]))
        if x > self._x[-1]:
            if self.extrap_method_ == ExtrapMethod.FLAT:
                return float(self._y[-1])
            return float(self._y[-1] + self._secants[-1] * (x - self._x[-1]))
        if x == self._x[-1]:
            return float(self._y[-1])

        i = int(np.searchsorted(self._x, x, side="right") - 1)
        h = self._h[i]
        t = (x - self._x[i]) / h
        a, b, c = self._interval_coefficients(i)
        return float(self._y[i] + a * t + b * t**2 + c * t**3)

    def integrate(self, start_x: float, end_x: float) -> float:
        start_x, end_x = float(start_x), float(end_x)
        if end_x < start_x:
            return -self.integrate(end_x, start_x)
        if end_x == start_x:
            return 0.0
        if self.length_ == 1:
            return float(self._y[0] * (end_x - start_x))

        total = 0.0
        left, right = start_x, end_x
        if left < self._x[0]:
            stop = min(right, self._x[0])
            width = stop - left
            if self.extrap_method_ == ExtrapMethod.FLAT:
                total += self._y[0] * width
            else:
                m = self._secants[0]
                total += self._y[0] * width + 0.5 * m * (
                    (stop - self._x[0])**2 - (left - self._x[0])**2
                )
            left = stop

        interior_left = max(left, self._x[0])
        interior_right = min(right, self._x[-1])
        if interior_right > interior_left:
            i = max(0, int(np.searchsorted(self._x, interior_left, side="right") - 1))
            while i < len(self._h) and self._x[i] < interior_right:
                lo = max(interior_left, self._x[i])
                hi = min(interior_right, self._x[i + 1])
                if hi > lo:
                    h = self._h[i]
                    t0, t1 = (lo - self._x[i]) / h, (hi - self._x[i]) / h
                    a, b, c = self._interval_coefficients(i)
                    total += self._y[i] * (hi - lo) + h * (
                        self._primitive(t1, a, b, c) - self._primitive(t0, a, b, c)
                    )
                i += 1

        if right > self._x[-1]:
            lo = max(left, self._x[-1])
            width = right - lo
            if self.extrap_method_ == ExtrapMethod.FLAT:
                total += self._y[-1] * width
            else:
                m = self._secants[-1]
                total += self._y[-1] * width + 0.5 * m * (
                    (right - self._x[-1])**2 - (lo - self._x[-1])**2
                )
        return float(total)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
