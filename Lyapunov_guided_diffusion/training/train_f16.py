from argparse import ArgumentParser
from copy import copy
import subprocess
import numpy as np

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
)
from neural_clbf.systems import F16


torch.multiprocessing.set_sharing_strategy("file_system")

controller_period = 0.05  # Higher control period for F16
simulation_dt = 0.01


def main(args):
    # Define the dynamics model with nominal parameters
    nominal_params = {
        "lag_error": 0.0,  # nominal value assumes no error in engine lag
    }
    
    print("Initializing F16 model...")
    dynamics_model = F16(
        nominal_params, dt=simulation_dt
    )
    
    # Initialize the DataModule with more focused initial conditions
    # Use narrower ranges to avoid extreme conditions that could cause numerical issues
    initial_conditions = [
        (450.0, 550.0),      # VT (airspeed)
        (-0.3, 0.3),         # alpha (angle of attack) - reduced range
        (-0.2, 0.2),         # beta (sideslip angle) - reduced range
        (-np.pi/4, np.pi/4), # phi (roll angle) - reduced range
        (-np.pi/6, np.pi/6), # theta (pitch angle) - reduced range
        (-np.pi/4, np.pi/4), # psi (yaw angle) - reduced range
        (-0.5, 0.5),         # P (roll rate) - reduced range
        (-0.5, 0.5),         # Q (pitch rate) - reduced range
        (-0.5, 0.5),         # R (yaw rate) - reduced range
        (-300.0, 300.0),     # pos_n - reduced range
        (-300.0, 300.0),     # pos_e - reduced range
        (200.0, 1000.0),     # altitude - narrower range to avoid extreme low altitudes
        (1.0, 8.0),          # pow (engine thrust dynamics) - avoid extremes
        (-5.0, 5.0),         # integrator state 1 - reduced range
        (-5.0, 5.0),         # integrator state 2 - reduced range
        (-5.0, 5.0),         # integrator state 3 - reduced range
    ]
    
    # Ensure initial_conditions has the correct length
    assert len(initial_conditions) == dynamics_model.n_dims, \
        f"Initial conditions has {len(initial_conditions)} elements but model has {dynamics_model.n_dims} dimensions"
    
    print("Creating data module...")
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=3,  # Reduced from 5
        trajectory_length=500,      # Reduced from 1000
        fixed_samples=5000,         # Reduced from 10000
        max_points=50000,           # Reduced from 100000
        val_split=0.1,
        batch_size=64,              # Smaller batch size for stability
        quotas={"safe": 0.4, "unsafe": 0.2, "goal": 0.2},
    )

    # Define fewer scenarios to reduce complexity
    print("Defining scenarios...")
    scenarios = []
    for lag_error in [0.0, 0.1]:  # Reduced number of scenarios and smaller values
        s = copy(nominal_params)
        s["lag_error"] = lag_error
        scenarios.append(s)
    
    print(f"Created {len(scenarios)} scenarios")

    # Create simpler experiments with fewer grid points
    print("Setting up experiments...")
    default_state = torch.zeros(dynamics_model.n_dims)
    default_state[F16.VT] = 500.0  # Set airspeed to a reasonable value
    default_state[F16.H] = 600.0   # Set altitude to a reasonable value
    
    # Altitude vs Pitch (critical for ground collision avoidance)
    # Using a smaller grid and narrower domain
    alt_pitch_experiment = CLFContourExperiment(
        "Altitude_vs_Pitch",
        domain=[(200.0, 1000.0), (-np.pi/6, np.pi/6)],
        n_grid=10,  # Reduced grid size
        x_axis_index=F16.H,
        y_axis_index=F16.THETA,
        x_axis_label="Altitude (ft)",
        y_axis_label="Pitch (rad)",
        plot_unsafe_region=True,
        default_state=default_state.clone(),
    )
    
    # Create a simplified experiment suite with just one experiment
    experiment_suite = ExperimentSuite([alt_pitch_experiment])

    # Initialize the controller with more conservative parameters
    print("Initializing CLBF controller...")
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=3,       
        clbf_hidden_size=128,        
        safe_level=10,
        controller_period=controller_period,
        num_init_epochs=5,          
        epochs_per_episode=100,     
        barrier=True,
        primal_learning_rate=5e-4,  # Lower learning rate for stability
        nominal_robustness_margin=0.5,  # Reduced from 1.0
    )

    # Initialize the logger and trainer with additional callbacks for stability
    current_git_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/f16/", name=f"commit_{current_git_hash}"
    )
    
    # Add callbacks for stability
    callbacks = [
        ModelCheckpoint(monitor="val_loss", save_top_k=2),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=20, mode="min"),
    ]
    
    # Create trainer with gradient clipping and reduced epochs
    trainer = pl.Trainer.from_argparse_args(
        args, 
        logger=tb_logger, 
        reload_dataloaders_every_epoch=True, 
        max_epochs=50,  # Reduced from 201
        gradient_clip_val=1.0,  # Add gradient clipping
        callbacks=callbacks,
        check_val_every_n_epoch=5,  # Validate less frequently
    )

    # Train
    print("Starting training...")
    # torch.autograd.set_detect_anomaly(True)  # Commented out to improve speed
    try:
        trainer.fit(clbf_controller)
    except Exception as e:
        import traceback
        print(f"Error during training: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc() 