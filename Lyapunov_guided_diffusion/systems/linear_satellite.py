"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List
from math import sqrt

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList
import numpy as np
import scipy.linalg


class LinearSatellite(ControlAffineSystem):
    """
    Represents a satellite through the linearized Clohessy-Wiltshire equations

    The system has state

        x = [x, y, z, xdot, ydot, zdot]

    representing the position and velocity of the chaser satellite, and it
    has control inputs

        u = [ux, uy, uz]

    representing the thrust applied in each axis. Distances are in km, and control
    inputs are measured in km/s^2.

    The task here is to get to the origin without leaving the bounding box [-5, 5] on
    all positions and [-1, 1] on velocities.

    The system is parameterized by
        a: the length of the semi-major axis of the target's orbit (e.g. 6871)
        ux_target, uy_target, uz_target: accelerations due to unmodelled effects and
                                         target control.
    """

    # Number of states and controls
    N_DIMS = 6
    N_CONTROLS = 3

    # State indices
    X = 0
    Y = 1
    Z = 2
    XDOT = 3
    YDOT = 4
    ZDOT = 5
    # Control indices
    UX = 0
    UY = 1
    UZ = 2

    # Constant parameters
    MU = 3.986e14  # Earth's gravitational parameter

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        use_l1_norm: bool = False,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )
        self.use_l1_norm = use_l1_norm

                # === Initialize LQR gain matrix K ===
        a = nominal_params["a"]
        n = np.sqrt(LinearSatellite.MU / a**3)

        # Define continuous-time A, B matrices
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n**2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n**2, 0, 0, 0]
        ])
        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        # Choose weighting matrices
        Q = np.diag([10, 10, 10, 1, 1, 1])
        R = np.diag([1, 1, 1])

        # Solve CARE and compute K
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P

        # Store as torch tensor
        self.K_lqr = torch.tensor(K, dtype=torch.float32)

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True
        # Make sure all needed parameters were provided
        valid = valid and "a" in params
        valid = valid and "ux_target" in params
        valid = valid and "uy_target" in params
        valid = valid and "uz_target" in params

        # Make sure all parameters are physically valid
        valid = valid and params["a"] > 0

        return valid

    @property
    def n_dims(self) -> int:
        return LinearSatellite.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return LinearSatellite.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[LinearSatellite.X] = 10.0
        upper_limit[LinearSatellite.Y] = 10.0
        upper_limit[LinearSatellite.Z] = 2.0
        upper_limit[LinearSatellite.XDOT] = 8
        upper_limit[LinearSatellite.YDOT] = 8
        upper_limit[LinearSatellite.ZDOT] = 8

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([5.0, 5.0, 5.0])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Stay within some maximum distance from the target
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : LinearSatellite.Z + 1].norm(dim=-1, p=order)
        # safe_mask.logical_and_(distance <= 1.0)

        # Stay at least some minimum distance from the target
        safe_mask.logical_and_(distance >= 0.75)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Maximum distance
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : LinearSatellite.Z + 1].norm(dim=-1, p=order)
        # unsafe_mask.logical_or_(distance >= 1.5)

        # Minimum distance
        unsafe_mask.logical_or_(distance <= 0.25)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        goal_mask = x[:, : LinearSatellite.Z + 1].norm(dim=-1, p=order) <= 0.5

        return goal_mask

    @property
    def u_eq(self) -> torch.Tensor:
        """
        Returns the control input for the system at the equilibrium point
        (which is the goal point at rest)
        
        returns:
            u_eq: the equilibrium control input as a torch.Tensor
        """
        # For equilibrium at the goal point, we need to compute the balance control
        goal = self.goal_point.squeeze()
        
        # Create a batch of size 1 with the goal state
        goal_batch = goal.unsqueeze(0)
        
        # Return the balancing control at the goal state
        return self.u_balance(goal_batch, self.nominal_params).squeeze()

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # Extract the needed parameters
        a = params["a"]
        ux_target = params["ux_target"]
        uy_target = params["uy_target"]
        uz_target = params["uz_target"]
        # Compute mean-motion
        n = sqrt(LinearSatellite.MU / a ** 3)
        # and state variables
        x_ = x[:, LinearSatellite.X]
        z_ = x[:, LinearSatellite.Z]
        xdot_ = x[:, LinearSatellite.XDOT]
        ydot_ = x[:, LinearSatellite.YDOT]
        zdot_ = x[:, LinearSatellite.ZDOT]

        # The first three dimensions just integrate the velocity
        f[:, LinearSatellite.X, 0] = xdot_
        f[:, LinearSatellite.Y, 0] = ydot_
        f[:, LinearSatellite.Z, 0] = zdot_

        # The last three use the CHW equations
        f[:, LinearSatellite.XDOT, 0] = 3 * n ** 2 * x_ + 2 * n * ydot_
        f[:, LinearSatellite.YDOT, 0] = -2 * n * xdot_
        f[:, LinearSatellite.ZDOT, 0] = -(n ** 2) * z_

        # Add perturbations
        f[:, LinearSatellite.XDOT, 0] += ux_target
        f[:, LinearSatellite.YDOT, 0] += uy_target
        f[:, LinearSatellite.ZDOT, 0] += uz_target

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # The control inputs are accelerations
        g[:, LinearSatellite.XDOT, LinearSatellite.UX] = 1.0
        g[:, LinearSatellite.YDOT, LinearSatellite.UY] = 1.0
        g[:, LinearSatellite.ZDOT, LinearSatellite.UZ] = 1.0

        return g

    def u_nominalsat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal LQR control input.

        args:
            x: bs x self.n_dims tensor of state
        returns:
            u_nom: bs x self.n_controls tensor of control input
        """
        # Ensure K_lqr shape: (n_controls, n_dims)
        K = self.K_lqr.to(x.device)
        u_nom = -torch.matmul(x, K.T)  # (bs, n_dims) x (n_dims, n_controls) = (bs, n_controls)
        return u_nom


    # def u_balance(self, x: torch.Tensor, params: Optional[Scenario] = None) -> torch.Tensor:
    #     """
    #     Compute the control input needed to balance the orbital dynamics forces.
    #     This calculates the control that would make f(x) + g(x)u = 0 for the
    #     velocity components.
        
    #     args:
    #         x: bs x self.n_dims tensor of state
    #         params: the model parameters used. If None, use nominal_params
    #     returns:
    #         u_balance: bs x self.n_controls tensor of balancing controls
    #     """
    #     # Use nominal parameters if none provided
    #     if params is None:
    #         params = self.nominal_params
            
    #     # Extract batch size and create tensor for result
    #     batch_size = x.shape[0]
    #     u_balance = torch.zeros((batch_size, self.n_controls)).type_as(x)
        
    #     # Extract the needed parameters
    #     a = params["a"]
    #     ux_target = params["ux_target"]
    #     uy_target = params["uy_target"]
    #     uz_target = params["uz_target"]
        
    #     # Compute mean-motion
    #     n = sqrt(LinearSatellite.MU / a ** 3)
        
    #     # Extract state variables
    #     x_ = x[:, LinearSatellite.X]
    #     z_ = x[:, LinearSatellite.Z]
    #     xdot_ = x[:, LinearSatellite.XDOT]
    #     ydot_ = x[:, LinearSatellite.YDOT]
        
    #     # Calculate balancing control to counter orbital dynamics
    #     u_balance[:, LinearSatellite.UX] = -(3 * n ** 2 * x_ + 2 * n * ydot_ + ux_target)
    #     u_balance[:, LinearSatellite.UY] = -(-2 * n * xdot_ + uy_target)
    #     u_balance[:, LinearSatellite.UZ] = -(-(n ** 2) * z_ + uz_target)
        
    #     return u_balance
    
