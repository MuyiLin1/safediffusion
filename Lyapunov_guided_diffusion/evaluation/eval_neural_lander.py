import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.systems import NeuralLander
import os

matplotlib.use('TkAgg')

def calculate_neural_lander_trajectories():
    # Create directory to save results
    if not os.path.exists("neural_lander_results"):
        os.makedirs("neural_lander_results")
    
    # Load the controller
    log_file = "input_your_log_file_here.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    
    # Get the dynamics model
    dynamics_model = neural_controller.dynamics_model
    
    # Define starting points
    start_states = [
        torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),  # Offset in x, y, z
        torch.tensor([-1.0, 1.0, 0.5, 0.0, 0.0, 0.0]),  # Offset in different quadrants
        torch.tensor([1.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        torch.tensor([-1.0, -1.0, 0.5, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),   # Just altitude
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # Just altitude
        torch.tensor([2.0, 2.0, 0.5, 0.0, 0.0, 0.0]),  # Offset in x, y, z
        torch.tensor([-2.0, -2.0, 0.5, 0.0, 0.0, 0.0]),  # Offset in different quadrants
        torch.tensor([2.0, -2.0, 0.5, 0.0, 0.0, 0.0]),
        torch.tensor([-2.0, 2.0, 0.5, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.7, 0.0, 0.0, 0.0]),   # Just altitude
        torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),   # Just altitude
        torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),   # Just altitude
    ]
    
    # Simulation parameters
    t_sim = 5.0
    dt = 0.1
    steps = int(t_sim / dt)
    neural_controller.clf_relaxation_penalty = 0
    goal = torch.zeros(dynamics_model.n_dims)
    
    trajectory_stats = []
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    safe_level = -0.3  # Unified safe level
    ax.set_zlim([safe_level - 0.1, 1])  # Expand z limits to show unsafe area

    # Draw the safe level plane
    # xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    # zz = np.full_like(xx, safe_level)
    # ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')
    # ax.text(0, 0, safe_level, "Safe Level", color='black', fontsize=15, ha='center')
    
    trajectory_colors = ['blue', 'green', 'pink', 'purple', 'orange']
    
    for i, start_state in enumerate(start_states):
        x_history = torch.zeros(steps+1, dynamics_model.n_dims)
        goal_distances = torch.zeros(steps+1)
        x_history[0] = start_state
        goal_distances[0] = torch.norm(start_state - goal).item()
        
        for t in range(steps):
            x_t = x_history[t].unsqueeze(0)
            u_t = neural_controller.forward(x_t)
            goal_distances[t] = torch.norm(x_history[t] - goal).item()
            x_dot = dynamics_model.closed_loop_dynamics(x_t, u_t)

            # adding white noise to the dynamics
            x_next = x_t + x_dot * dt + torch.randn_like(x_dot) * 0.01
            x_history[t+1] = x_next.squeeze(0)
        
        goal_distances[steps] = torch.norm(x_history[steps] - goal).item()
        ground_collision = any(x_history[t, NeuralLander.PZ].item() < safe_level for t in range(steps+1))
        
        trajectory_stats.append({
            'trajectory': i+1,
            'start_state': start_state.detach().numpy(),
            'ground_collision': ground_collision,
            'initial_distance': goal_distances[0].item(),
            'final_distance': goal_distances[-1].item(),
            'distance_reduction': goal_distances[0].item() - goal_distances[-1].item(),
            'distance_reduction_percent': 100 * (1 - goal_distances[-1].item() / goal_distances[0].item()) if goal_distances[0].item() > 0 else 0
        })
        
        x_pos = [x_history[t, NeuralLander.PX].detach().item() for t in range(steps+1)]
        y_pos = [x_history[t, NeuralLander.PY].detach().item() for t in range(steps+1)]
        z_pos = [x_history[t, NeuralLander.PZ].detach().item() for t in range(steps+1)]
        
        # Plot trajectory with unsafe parts in red
        color = trajectory_colors[i % len(trajectory_colors)]
        segment_x, segment_y, segment_z = [], [], []
        current_color = None
        
        for t in range(steps + 1):
            z_val = z_pos[t]
            is_safe = z_val >= safe_level
            segment_color = color if is_safe else 'red'
            
            if current_color is None:
                current_color = segment_color
            
            if segment_color != current_color:
                ax.plot(segment_x, segment_y, segment_z, color=current_color, linewidth=2)
                segment_x, segment_y, segment_z = [], [], []
                current_color = segment_color
            
            segment_x.append(x_pos[t])
            segment_y.append(y_pos[t])
            segment_z.append(z_pos[t])
        
        if segment_x:
            ax.plot(segment_x, segment_y, segment_z, color=current_color, linewidth=2)
        
        ax.scatter(x_pos[0], y_pos[0], z_pos[0], marker='o', s=100, color=color)
        ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], marker='x', s=100, color='black')
    
    ax.scatter(0, 0, 0, marker='*', s=200, color='gold', label='Goal')
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.set_xlabel('X Position', fontsize=20, labelpad=15)
    ax.set_ylabel('Y Position', fontsize=20, labelpad=15)
    ax.set_zlabel('Z Position', fontsize=20, labelpad=15)
    ax.legend(fontsize=18)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5])
    ax.set_zticks([safe_level, 0, 0.25, 0.5, 0.75, 1])
    ax.view_init(elev=20, azim=110)
    
    plt.savefig("neural_lander_results/trajectories_3d.png")
    plt.tight_layout()
    plt.show()
    
    print("\nTrajectory Statistics:")
    print("=================")
    for stat in trajectory_stats:
        status = "COLLISION" if stat['ground_collision'] else "SAFE"
        print(f"Trajectory {stat['trajectory']}: {status}")
        print(f"  Distance to goal: {stat['initial_distance']:.4f} m → {stat['final_distance']:.4f} m")
        print(f"  Distance reduction: {stat['distance_reduction']:.4f} m ({stat['distance_reduction_percent']:.2f}%)")
        print("-----------------")

def plot_neural_lander_clf():
    # Load the checkpoint file
    log_file = "your_log_file_here.ckpt"
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    
    # Calculate and plot trajectories only
    calculate_neural_lander_trajectories()


if __name__ == "__main__":
    plot_neural_lander_clf()