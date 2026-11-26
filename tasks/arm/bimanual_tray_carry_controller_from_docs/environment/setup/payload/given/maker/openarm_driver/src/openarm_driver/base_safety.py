# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base safety module - generic framework for collision and workspace checking.

This module provides abstract base classes and protocols that users can
implement with their own checking logic (e.g., MuJoCo collision meshes,
custom FK, point cloud checks, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class CheckResult:
    """Result of a safety check."""

    is_safe: bool
    force_stop: bool = False
    fixed_joint_positions: ArrayLike | None = None
    message: str = ""
    check_type: str | None = None  # "collision", "workspace", "velocity", etc.
    details: dict[str, Any] = field(default_factory=dict)


class Checker(ABC):
    """Abstract base class for safety checking.

    Subclass this to implement custom checking logic:
    - FK-based workspace bounds
    - Collision mesh intersection (MuJoCo, PyBullet, etc.)
    - Point cloud distance checks
    - Neural network predictions
    - etc.
    """

    @abstractmethod
    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Check if joint positions are safe.

        Args:
            joint_positions: Target joint positions to check
            **kwargs: Additional context (velocities, current state, etc.)

        Returns:
            CheckResult with safety status and details

        """
        pass

    def reset(self):
        """Reset any internal state. Override if needed."""
        pass


class NullChecker(Checker):
    """No-op checker that always returns safe. Use as placeholder."""

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        return CheckResult(is_safe=True, message="No check performed")


class CompositeChecker(Checker):
    """Combines multiple checkers into a single checker.

    Runs all checkers and returns the first failure, or success if all pass.
    All checkers run sequentially. If a failure marked as 'force_stop' occurs,
    the result is returned immediately. If a failure that is not marked as
    'force_stop' occurs, the checks continue using the 'fixed_joint_positions'.
    """

    def __init__(self, checkers: list[Checker] | None = None):
        """Init parameters.

        Args:
            checkers: The underlying checkers.

        """
        self.checkers: list[Checker] = checkers if checkers is not None else []

    def add(self, checker: Checker) -> "CompositeChecker":
        """Add a checker. Returns self for chaining."""
        self.checkers.append(checker)
        return self

    def remove(self, checker: Checker) -> bool:
        """Remove a checker. Returns True if found and removed."""
        try:
            self.checkers.remove(checker)
            return True
        except ValueError:
            return False

    def clear(self):
        """Remove all checkers."""
        self.checkers.clear()

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        not_safe_results = []
        is_safe = True
        for checker in self.checkers:
            result = checker.check(joint_positions, **kwargs)
            if not result.is_safe:
                not_safe_results.append(result)
                if result.force_stop:
                    return result
                if result.fixed_joint_positions is not None:
                    is_safe = False
                    joint_positions = result.fixed_joint_positions

        if is_safe:
            return CheckResult(
                is_safe=True,
                message=f"All {len(self.checkers)} checks passed",
            )
        else:
            return CheckResult(
                is_safe=False,
                force_stop=False,
                check_type=not_safe_results[-1].check_type,
                fixed_joint_positions=joint_positions,
                message=not_safe_results[-1].message,
            )

    def reset(self):
        """Reset internal state."""
        for checker in self.checkers:
            checker.reset()


class ConditionalChecker(Checker):
    """Wraps a checker with a condition function.

    Only runs the underlying check if the condition returns True.
    """

    def __init__(
        self,
        checker: Checker,
        condition: callable,
        skip_message: str = "Check skipped by condition",
    ):
        """Init parameters.

        Args:
            checker: The underlying checker to conditionally run
            condition: Function (joint_positions, **kwargs) -> bool
            skip_message: Message when check is skipped

        """
        self.checker = checker
        self.condition = condition
        self.skip_message = skip_message

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        if self.condition(joint_positions, **kwargs):
            return self.checker.check(joint_positions, **kwargs)
        return CheckResult(is_safe=True, message=self.skip_message)

    def reset(self):
        """Reset internal state."""
        self.checker.reset()


class CachedChecker(Checker):
    """Caches the last check result if position hasn't changed.

    Useful for expensive checks (mesh collision, neural nets).
    """

    def __init__(self, checker: Checker, tolerance: float = 1e-6):
        """Init parameters.

        Args:
            checker: The underlying checker.
            tolerance: The absolute tolerance of position diff.

        """
        self.checker = checker
        self.tolerance = tolerance
        self._last_positions: np.ndarray | None = None
        self._last_result: CheckResult | None = None

    def check(self, joint_positions: ArrayLike, **kwargs) -> CheckResult:
        """Run check."""
        positions = np.asarray(joint_positions, dtype=float)

        if (
            self._last_positions is not None
            and self._last_result is not None
            and np.allclose(positions, self._last_positions, atol=self.tolerance)
        ):
            return self._last_result

        result = self.checker.check(positions, **kwargs)
        self._last_positions = positions.copy()
        self._last_result = result
        return result

    def reset(self):
        """Reset internal state."""
        self._last_positions = None
        self._last_result = None
        self.checker.reset()
