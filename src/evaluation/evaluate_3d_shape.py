from evaluation.evaluate import Evaluate
import pickle
import plotly.graph_objects as go
from agent.utils.dynamical_system_operations import denormalize_state
import agent.utils.polar as polar

from evaluation.utils.similarity_measures import get_chamfer_distance, get_spurious_attractors_shape
from evaluation.utils.saving import save_stats_txt_continuous
import torch
import numpy as np



class Evaluate3D_shape(Evaluate):
    """
    Class for evaluating three-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.show_plotly = params.show_plotly

        #self.shape = denormalize_state(data['shape_attractors'][0], self.x_max, self.x_min)
        self.shape = data['shape_attractors'][0]
        #self.hard_neg = denormalize_state(data['shape_attractors'][1], self.x_max, self.x_min)
        self.hard_neg = data['shape_attractors'][1]
        self.attractor_shape_orientation = data['orientation_parametrization_shape']

        self.print_demos = params.print_demos
        self.imitation_component = params.imitation_loss_weight

        self.chamfer = []
        self.latent_orientation_error = []
        self.spurious_attractors = []

        # just put huge number for checking if i obtained a best model
        self.best_chamfer = 10000
        self.best_oriented_error = 10000
        self.best_model = False



    def compute_quali_eval(self, sim_results, attractor, primitive_id, iteration):
        """
        Computes qualitative results
        """
        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pickle'

        attractor = sim_results['visited states grid'][-1, :, :]

        self.plot_DS_plotly(sim_results['visited states demos'], attractor, save_path)

        # to check, what if we have other val in sim_results?
        # can i select a grid and let the point being evaluated since that?
        # do i have any info on the velocity?

        return True

    def compute_quali_eval_without_imit(self, sim_results, primitive_id, iteration):

        save_path = self.learner.save_path + 'images/' + 'primitive_%i_iter_%i' % (primitive_id, iteration) + '.pickle'


    def compute_diffeo_quali_eval(self, sim_results, sim_results_latent, primitive_id, iteration):
        # Not implemented
        return False

    def plot_DS_plotly(self, visited_states, attractor, save_path):
        plot_data = []


        plot_data.append(
            go.Scatter3d(
                x = attractor[:, 2],
                y = attractor[:, 0],
                z = attractor[:, 1],
                marker=go.scatter3d.Marker(size=3, color='green'),
                opacity=0.6,
                mode='markers'
            )
        )

        plot_data.append(
            go.Scatter3d(
                x=self.shape[:, 2],
                y=self.shape[:, 0],
                z=self.shape[:, 1],
                mode='lines',  # You can also use 'lines' or 'markers'
                line=dict(color='blue', width=4),
            )
        )
        plot_data.append(
            go.Scatter3d(
                x = self.hard_neg[:, 2],
                y = self.hard_neg[:, 0],
                z = self.hard_neg[:, 1],
                marker=go.scatter3d.Marker(size=1, color='red'),
                opacity=0.1,
                mode='markers'
            )
        )

        if self.print_demos :
            for i in range(self.n_trajectories):
                plot_data.append(
                    go.Scatter3d(
                        x=self.demonstrations_eval[i][2],  # TODO: match corresponding axes using a parameter
                        y=self.demonstrations_eval[i][0],
                        z=self.demonstrations_eval[i][1],
                        marker=go.scatter3d.Marker(size=3, color='red'),
                        opacity=0.5,
                        mode='markers',
                        name='demonstration %i' % i,
                    )
                )

                denorm_visited_states = denormalize_state(visited_states, self.x_min, self.x_max)
                plot_data.append(
                    go.Scatter3d(
                        x=denorm_visited_states[:, i, 2],
                        y=denorm_visited_states[:, i, 0],
                        z=denorm_visited_states[:, i, 1],
                        marker=go.scatter3d.Marker(size=3, color='blue'),
                        opacity=0.5,
                        mode='markers',
                        name='CONDOR %i' % i,
                    )
                )

        layout = go.Layout(autosize=True,
                           scene=dict(
                               xaxis_title='x (m)',
                               yaxis_title='y (m)',
                               zaxis_title='z (m)'),
                           margin=dict(l=65, r=50, b=65, t=90),
                           showlegend=True,
                           font=dict(family='Time New Roman', size=15))
        fig = go.Figure(data=plot_data, layout=layout)

        plot_data = {'3D_plot': fig}

        # Save
        print('Saving image data to %s...' % save_path)
        pickle.dump(plot_data, open(save_path, 'wb'))

        if self.show_plotly:
            fig.show()

    def plot_DS_plotly_old(self, visited_states, attractor, save_path):
        """
        Plots demonstrations and simulated trajectories when starting from demos initial states
        """
        plot_data = []
        for i in range(self.n_trajectories):
            # Plot datasets


            marker_data_demos = go.Scatter3d(
                x=self.demonstrations_eval[i][2],  # TODO: match corresponding axes using a parameter
                y=self.demonstrations_eval[i][0],
                z=self.demonstrations_eval[i][1],
                marker=go.scatter3d.Marker(size=3, color='red'),
                opacity=0.5,
                mode='markers',
                name='demonstration %i' % i,
            )
            plot_data.append(marker_data_demos)

            # Plot network executions
            denorm_visited_states = denormalize_state(visited_states, self.x_min, self.x_max)
            marker_data_executed = go.Scatter3d(
                x=denorm_visited_states[:, i, 2],
                y=denorm_visited_states[:, i, 0],
                z=denorm_visited_states[:, i, 1],
                marker=go.scatter3d.Marker(size=3, color='blue'),
                opacity=0.5,
                mode='markers',
                name='CONDOR %i' % i,
            )
            plot_data.append(marker_data_executed)

        #attractor = denormalize_state(attractor, self.x_min, self.x_max)
        attractors_demo = denormalize_state(visited_states[-1, :, :], self.x_min, self.x_max)

        marker_data_attractors = go.Scatter3d(
            x=attractor[:, 2],
            y=attractor[:, 0],
            z=attractor[:, 1],
            marker=go.scatter3d.Marker(size=3, color='green'),
            opacity=0.5,
            mode='markers'
        )

        marker_data_attractors_demo = go.Scatter3d(
            x=attractors_demo[:, 2],
            y=attractors_demo[:, 0],
            z=attractors_demo[:, 1],
            marker=go.scatter3d.Marker(size=3, color='green'),
            opacity=0.5,
            mode='markers'
        )
        plot_data.append(marker_data_attractors_demo)
        plot_data.append(marker_data_attractors)

        """
        reference_shape = go.Scatter3d(
                x=self.shape_attractors[:, 2],
                y=self.shape_attractors[:, 0],
                z=self.shape_attractors[:, 1],
                marker=go.scatter3d.Marker(size=3, color='red'),
                opacity=1.0,
                mode='markers'
        )
        plot_data.append(reference_shape)
        """

        layout = go.Layout(autosize=True,
                           scene=dict(
                               xaxis_title='x (m)',
                               yaxis_title='y (m)',
                               zaxis_title='z (m)'),
                           margin=dict(l=65, r=50, b=65, t=90),
                           showlegend=True,
                           font=dict(family='Time New Roman', size=15))
        fig = go.Figure(data=plot_data, layout=layout)

        plot_data = {'3D_plot': fig}

        # Save
        print('Saving image data to %s...' % save_path)
        pickle.dump(plot_data, open(save_path, 'wb'))

        if self.show_plotly:
            fig.show()

        return True

    def get_shape_metrics(self, attractors):
        # attractors = denormalize_state(attractors, self.x_min, self.x_max)
        # shape = denormalize_state(self.shape, self.x_min, self.x_max)

        # Chamfer metrics:
        #   measure of similarity between the target shape and the generated one
        chamfer = get_chamfer_distance(attractors, self.shape)
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
            x_t = torch.FloatTensor(self.shape).cuda()
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
