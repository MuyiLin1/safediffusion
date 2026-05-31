"""
Planar (2‑D) linearized Clohessy‑Wiltshire model for a chaser satellite.

State  : x = [px, py, vx, vy]         (km , km , km/s , km/s)
Control: u = [fx, fy]                 (km/s² in radial & along‑track)

安全/危险区域 (见论文图 1) ——  
  • X_safe   = { 4 ≤ r ≤ 8 } ∪ { py ≤ –|px|  ∧  r ≤ 4 }  
  • X_unsafe = { r ≥ 9 } ∪ { py ≥ –|px|  ∧  r ≤ 3 }  
  • X_goal   = { py ≤ –|px|  ∧  r ≤ 0.5 }      # 贴近目标且保持 LOS
"""

from typing import Tuple, Optional, List
from math import sqrt

import torch
from .control_affine_system import ControlAffineSystem           # noqa: F401
from neural_clbf.systems.utils import Scenario, ScenarioList      # noqa: F401


class LinearSatellite2D(ControlAffineSystem):
    """Planar satellite model (linear CW)."""

    # ──────────────────────── 维度定义 ────────────────────────
    N_DIMS = 4
    N_CONTROLS = 2

    PX, PY, VX, VY = range(4)
    FX, FY = range(2)

    MU = 3.986e14  # m³/s²

    # ──────────────────────── 初始化 ────────────────────────
    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    # ──────────────────────── 参数检查 ────────────────────────
    def validate_params(self, params: Scenario) -> bool:
        required = {"a", "fx_target", "fy_target"}
        return required.issubset(params.keys()) and params["a"] > 0

    # ──────────────────────── 元信息 ────────────────────────
    @property
    def n_dims(self) -> int:
        return self.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []  # all Cartesian

    @property
    def n_controls(self) -> int:
        return self.N_CONTROLS

    # ──────────────────────── 状态 / 控制范围 ────────────────────────
    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper = torch.tensor([2.0, 2.0, 1.0, 1.0])
        return upper, -upper

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper = torch.tensor([1.0, 1.0])
        return upper, -upper

    # ──────────────────────── 区域 mask ────────────────────────
    def _polar_radius(self, x: torch.Tensor) -> torch.Tensor:
        """2‑norm distance to target."""
        return torch.linalg.norm(x[:, :2], dim=-1)

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Safe when inside LOS sector or ring 4–8 km."""
        r = self._polar_radius(x)
        px, py = x[:, self.PX], x[:, self.PY]
        wedge = py <= -torch.abs(px)
        ring = (r >= 4.0) & (r <= 8.0)
        sector = wedge & (r <= 4.0)
        return ring | sector

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Unsafe when too far (r≥9) or inside forbidden near‑origin region."""
        r = self._polar_radius(x)
        px, py = x[:, self.PX], x[:, self.PY]
        outside = r >= 9.0
        near_origin_bad = (~(py <= -torch.abs(px))) & (r <= 3.0)
        return outside | near_origin_bad

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Goal: close docking zone inside LOS sector (r≤0.5)."""
        r = self._polar_radius(x)
        px, py = x[:, self.PX], x[:, self.PY]
        return (py <= -torch.abs(px)) & (r <= 0.5)

    # ──────────────────────── 动力学 f(x) ────────────────────────
    def _f(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        bs = x.shape[0]
        f = torch.zeros(bs, self.n_dims, 1, dtype=x.dtype, device=x.device)

        a = params["a"]
        fx_t = params["fx_target"]
        fy_t = params["fy_target"]
        n = sqrt(self.MU / a**3)

        # position derivatives
        f[:, self.PX, 0] = x[:, self.VX]
        f[:, self.PY, 0] = x[:, self.VY]

        # velocity derivatives (CW planar)
        f[:, self.VX, 0] = 3 * n**2 * x[:, self.PX] + 2 * n * x[:, self.VY] + fx_t
        f[:, self.VY, 0] = -2 * n * x[:, self.VX] + fy_t

        return f

    # ──────────────────────── 动力学 g(x) ────────────────────────
    def _g(self, x: torch.Tensor, params: Scenario) -> torch.Tensor:
        bs = x.shape[0]
        g = torch.zeros(bs, self.n_dims, self.n_controls, dtype=x.dtype, device=x.device)
        g[:, self.VX, self.FX] = 1.0
        g[:, self.VY, self.FY] = 1.0
        return g
