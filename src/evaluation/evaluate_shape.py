import matplotlib.pyplot as plt
from evaluation.evaluate import Evaluate
from matplotlib.patches import Ellipse
import numpy as np
from agent.utils.dynamical_system_operations import denormalize_state
import agent.utils.polar as polar

from evaluation.utils.similarity_measures import get_chamfer_distance, get_spurious_attractors_shape
from evaluation.utils.saving import save_stats_txt_continuous
import torch


class EvaluateShape(Evaluate):
    """
    Class for evaluating first-order two-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.attractor_shape = data['shape_attractors'][0]
        self.diffeo_comparison_length = params.diffeo_comparison_length
        self.diffeo_comparison_subsample = params.diffeo_comparison_subsample
        self.print_demos = params.print_demos
        self.imitation_component = params.imitation_loss_weight

        self.chamfer = []
        self.latent_orientation_error = []
        self.spurious_attractors = []

        # just put huge number for checking if i obtained a best model
        self.best_chamfer = 10000
        self.best_oriented_error = 10000
        self.best_model = False

        self.shape = denormalize_state(data['shape_attractors'][0], self.x_min, self.x_max)
        self.attractor_shape_orientation = data['orientation_parametrization_shape']



    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration):
        """
        Computes qualitative results
        """
        # Get velocities
        vel = self.get_vector_field(sim_results['initial states grid'], primitive_id)

        # Plot
        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pdf'
        self.plot_dynamical_system(sim_results, vel, attractor, primitive_id,
                                   title='Dynamical System',
                                   save_path=save_path)
        return True

    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        """
        Computes qualitative results diffeomorphism
        """
        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '_latent.pdf'

        self.plot_diffeo_comparison(sim_results['visited states grid'],
                                    sim_results_latent['visited states grid'],
                                    self.diffeo_comparison_length,
                                    self.diffeo_comparison_subsample,
                                    primitive_id,
                                    title='Diffeomorphism Comparison',
                                    save_path=save_path)
        return True

    def plot_dynamical_system(self, sim_results, vel, attractor, primitive_i, title, save_path,
                              show=False, obstacles=None):
        """
        Plots demonstrations, simulated trajectories, attractor and vector field
        """
        # Update plot params
        plt.rcParams.update({'font.size': 14,
                             'figure.figsize': (8, 9)})

        # Obstacle avoidance
        if obstacles is not None:
            obstacle_avoidance = True
            ax = plt.gca()
        else:
            obstacle_avoidance = False

        # Get denormalized states equally-spaced grid
        grid_x1 = denormalize_state(sim_results['grid'][0], self.x_min[0], self.x_max[0])
        grid_x2 = denormalize_state(sim_results['grid'][1], self.x_min[1], self.x_max[1])

        # Plot vector field
        plt.streamplot(grid_x1, grid_x2, vel[:, :, 0], vel[:, :, 1],
                       linewidth=0.5, density=2, arrowstyle='fancy',
                       arrowsize=1, color='black', cmap=plt.cm.Greys)

        # Plot contour with norm of the velocities
        norm_vel = np.linalg.norm(vel, axis=2)  # compute norm velocities
        CS = plt.contourf(grid_x1, grid_x2, norm_vel, cmap='viridis', levels=50)
        cbar = plt.colorbar(CS, location='bottom')
        cbar.ax.set_xlabel('speed (mm/s)')

        if self.print_demos:
            # Plot demonstrations (we need loop because they could have different lengths)
            for i in range(self.n_trajectories):
                if [self.primitive_ids == primitive_i][0][i]:
                    plt.scatter(self.demonstrations_eval[i][0], self.demonstrations_eval[i][1], color='white', alpha=0.5)

            # Plot trajectories that start from the same points as the demonstrations
            plt.plot(denormalize_state(sim_results['visited states demos'][:, :self.n_trajectories, 0],
                                       self.x_min[0], self.x_max[0]),
                     denormalize_state(sim_results['visited states demos'][:, :self.n_trajectories, 1],
                                       self.x_min[1], self.x_max[1]),
                     linewidth=4, color='red', zorder=11)

        # Plot shape
        plt.scatter(self.shape[:, 0], self.shape[:, 1], color='white', linewidth=2)
        # Plot attractors
        #if not self.latent_dynamic_system_type == 'limit cycle':
        if self.imitation_component > 0 :
            plt.scatter(attractor[:, 0], attractor[:, 1], linewidth=4, color='blue', zorder=12)
        else:
            plt.scatter(attractor[:, 0], attractor[:, 1], linewidth=4, color='red', zorder=12)


        # Plot ellipse when obstacle avoidance
        if obstacle_avoidance:
            for i in range(len(obstacles['centers'])):
                ellipse = Ellipse(xy=(obstacles['centers'][i][0], obstacles['centers'][i][1]),
                                  width=obstacles['axes'][i][0] * 2,
                                  height=obstacles['axes'][i][1] * 2,
                                  edgecolor='mlearnera',
                                  fc='mlearnera',
                                  lw=2,
                                  zorder=10)
                ax.add_artist(ellipse)

        # Plot details/info
        plt.xlim([self.x_min[0], self.x_max[0]])
        plt.ylim([self.x_min[1], self.x_max[1]])
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(title)

        # Save
        print('Saving image to %s...' % save_path)
        plt.savefig(save_path, bbox_inches='tight')

        if show:
            plt.show()

        # Close
        plt.clf()
        plt.close()

        return True

    def plot_diffeo_comparison(self, visited_states, visited_states_latent, trajectory_length, subsample_rate,
                               primitive_id, title, save_path, show=False):
        """
        Plots diffeomorphism comparison
        """
        # Update plot params
        plt.rcParams.update({'font.size': 14,
                             'figure.figsize': (8, 8)})

        # Plot demonstrations (we need loop cause they could have different lengths)
        for i in range(self.n_trajectories):
            if [self.primitive_ids == primitive_id][0][i]:
                plt.scatter(self.demonstrations_eval[i][0], self.demonstrations_eval[i][1], color='cyan')

        # Plot trajectories that start from the same points as the demonstrations
        plt.plot(denormalize_state(visited_states[:trajectory_length, 1::subsample_rate, 0], self.x_min[0], self.x_max[0]),
                 denormalize_state(visited_states[:trajectory_length, 1::subsample_rate, 1], self.x_min[1], self.x_max[1]),
                 linewidth=15, color='black', linestyle='-')

        plt.plot(denormalize_state(visited_states_latent[:trajectory_length, 1::subsample_rate, 0], self.x_min[0], self.x_max[0]),
                 denormalize_state(visited_states_latent[:trajectory_length, 1::subsample_rate, 1], self.x_min[1], self.x_max[1]),
                 linewidth=10, color='red', linestyle='-')

        # Plot details/info
        plt.xlim([self.x_min[0], self.x_max[0]])
        plt.ylim([self.x_min[1], self.x_max[1]])
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title(title)
        
        # Show and save
        print('Saving image to %s...' % save_path)
        plt.savefig(save_path)

        if show:
            plt.show()

        # Close
        plt.clf()
        plt.close()

        return True


    def get_shape_metrics(self, attractors):
        #attractors = denormalize_state(attractors, self.x_min, self.x_max)
        shape = denormalize_state(self.attractor_shape, self.x_min, self.x_max)

        # Chamfer metrics:
        #   measure of similarity between the target shape and the generated one
        chamfer = get_chamfer_distance(attractors, shape)
        self.chamfer.append(chamfer)

        # Spurios attractor: N of attractor (end points) for which the distance from the reference shape
        #                    is larger of a threshold
        spurious_attractor_percentage = get_spurious_attractors_shape(attractors,
                                                                      self.shape,
                                                                      self.fixed_point_iteration_thr)
        # convert to percentage
        spurious_attractor_percentage = spurious_attractor_percentage / len(attractors)
        self.spurious_attractors.append(spurious_attractor_percentage)

        # latent error on the orientation:
        with torch.no_grad():
            x_t = torch.FloatTensor(self.attractor_shape).cuda()
            y_t = self.learner.model.encoder(x_t, torch.zeros(1))
            y_t_polar = polar.euclidean_to_polar(y_t)

        theta = y_t_polar[:, 1].cpu().numpy()
        distance = np.abs(theta - self.attractor_shape_orientation)
        latent_orientation_error = np.min(np.stack((distance, 2*np.pi - distance), axis=1), axis=1)
        latent_orientation_error = np.linalg.norm(latent_orientation_error, ord=1)
        self.latent_orientation_error.append(latent_orientation_error)

        shape_metrics = {
            "chamfer": chamfer,
            "spurious_percentage":  spurious_attractor_percentage,
            "latent orientation error": latent_orientation_error
        }

        return shape_metrics

    # overide
    def get_accuracy_metrics(self, visited_states, demonstrations_eval, max_trajectory_length, eval_indexes):
        return {}

    def get_stability_metrics(self, attractor, goal):
        return self.get_shape_metrics(attractor)

    def select_best_model(self, metrics_stab):
        self.best_model = False

        if metrics_stab["chamfer"] < self.best_chamfer:
            self.best_model = True
            self.best_chamfer = metrics_stab['chamfer']

    def compute_quanti_eval(self, sim_results, attractors, primitive_id):
       # metrics_acc = super().get_stability_metrics(sim_results['visited states demos'],
       #                                             self.demonstrations_eval,
       #                                             self.max_trajectory_length,
       #                                             self.eval_indexes)

        metrics_stab = self.get_stability_metrics(attractors, None)

        self.select_best_model(metrics_stab)

        # TODO find a way for avoid to pass the empty dict : it comes form the accuracy metrics
        return {}, metrics_stab


    def save_progress(self, save_path, i, model, writer):

        np.save(save_path + "stats/chamfer", self.chamfer)
        save_stats_txt_continuous(save_path,
                                  self.chamfer,
                                  self.latent_orientation_error,
                                  self.spurious_attractors,
                                  iteration=i)
        # save txt with info
        if self.best_model:
            torch.save(model.state_dict(), save_path + "model")
