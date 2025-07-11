from evaluation.evaluate import Evaluate
import pickle
import plotly.graph_objects as go
from agent.utils.dynamical_system_operations import denormalize_state


class Evaluate3D(Evaluate):
    """
    Class for evaluating three-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.show_plotly = params.show_plotly
        if params.shaped_attractors_form:
            self.shape_attractors = denormalize_state(data['shape_attractors'][0], self.x_max, self.x_min)


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

        attractor = denormalize_state(attractor, self.x_min, self.x_max)
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
