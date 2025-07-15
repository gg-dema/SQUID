
from evaluation.evaluate import Evaluate
from scipy.spatial.transform import Rotation as R
import numpy as np
import pickle
import plotly.graph_objects as go
from agent.utils.dynamical_system_operations import denormalize_state


class Evaluate7D_frame(Evaluate):
    """
    Class for evaluating three-dimensional dynamical systems
    """
    def __init__(self, learner, data, params, verbose=True):
        super().__init__(learner, data, params, verbose=verbose)

        self.show_plotly = params.show_plotly
        self.goals = data['goals']

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

        denorm_visited_states = denormalize_state(visited_states, self.x_min, self.x_max)

        for i in range(self.n_trajectories):
            # Demonstration
            plot_data.append(go.Scatter3d(
                x=self.demonstrations_eval[i][2],
                y=self.demonstrations_eval[i][0],
                z=self.demonstrations_eval[i][1],
                marker=go.scatter3d.Marker(size=3, color='red'),
                opacity=0.5,
                mode='markers',
                name=f'demonstration {i}',
            ))

            # Executed trajectory
            plot_data.append(go.Scatter3d(
                x=denorm_visited_states[:, i, 2],
                y=denorm_visited_states[:, i, 0],
                z=denorm_visited_states[:, i, 1],
                marker=go.scatter3d.Marker(size=3, color='blue'),
                opacity=0.5,
                mode='markers',
                name=f'CONDOR {i}',
            ))

        attractor = denormalize_state(attractor, self.x_min, self.x_max)
        attractors_demo = denormalize_state(visited_states[-1, :, :], self.x_min, self.x_max)
        attractor_pos = attractor[:, :3]
        attractor_ori = attractor[:, 3:]
        axis_len = 0.01  # fixed small axis length

        for goal_id in range(attractor.shape[0]):
            r = R.from_quat([
                attractor_ori[goal_id, 1],
                attractor_ori[goal_id, 2],
                attractor_ori[goal_id, 3],
                attractor_ori[goal_id, 0],
            ])
            rot_matrix = r.as_matrix()

            axes = {
                'x': rot_matrix @ np.array([axis_len, 0, 0]),
                'y': rot_matrix @ np.array([0, axis_len, 0]),
                'z': rot_matrix @ np.array([0, 0, axis_len]),
            }
            colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

            for axis_name, vec in axes.items():
                plot_data.append(go.Scatter3d(
                    x=[attractor_pos[goal_id, 2], attractor_pos[goal_id, 2] + vec[0]],
                    y=[attractor_pos[goal_id, 0], attractor_pos[goal_id, 0] + vec[1]],
                    z=[attractor_pos[goal_id, 1], attractor_pos[goal_id, 1] + vec[2]],
                    mode='lines',
                    line=dict(color=colors[axis_name], width=3),
                ))

        plot_data.append(go.Scatter3d(
            x=attractors_demo[:, 2],
            y=attractors_demo[:, 0],
            z=attractors_demo[:, 1],
            marker=go.scatter3d.Marker(size=4, color='green'),
            opacity=0.8,
            mode='markers',
            name='final positions'
        ))

        # Determine axis limits based on data to avoid zoom-out issues
        all_x = np.concatenate([denorm_visited_states[:, :, 2].flatten(), attractor_pos[:, 2]])
        all_y = np.concatenate([denorm_visited_states[:, :, 0].flatten(), attractor_pos[:, 0]])
        all_z = np.concatenate([denorm_visited_states[:, :, 1].flatten(), attractor_pos[:, 1]])

        def axis_range(arr, margin=0.1):
            min_v, max_v = arr.min(), arr.max()
            range_v = max_v - min_v
            return [min_v - margin * range_v, max_v + margin * range_v]

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='x (m)', range=axis_range(all_x)),
                yaxis=dict(title='y (m)', range=axis_range(all_y)),
                zaxis=dict(title='z (m)', range=axis_range(all_z)),
            ),
            margin=dict(l=65, r=50, b=65, t=90),
            showlegend=True,
            font=dict(family='Times New Roman', size=15)
        )

        fig = go.Figure(data=plot_data, layout=layout)

        # Save
        print(f'Saving image data to {save_path}...')
        pickle.dump({'3D_plot': fig}, open(save_path, 'wb'))

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
            # this should be transformed in the end point of the demonstrations
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

        attractor_pos = attractor[:, :3]
        attractor_ori = attractor[:, 3:]
        axis_len = 0.001
        for end_point in range(attractor.shape[0]):

            r = R.from_quat([
                attractor_ori[end_point, 1],
                attractor_ori[end_point, 2],
                attractor_ori[end_point, 3],
                attractor_ori[end_point, 0],
                ])  # SciPy expects (x, y, z, w)

            rot_matrix = r.as_matrix()

            # Unit vectors
            axes = {
                'x': rot_matrix @ np.array([end_point, 0, 0]),
                'y': rot_matrix @ np.array([0, end_point, 0]),
                'z': rot_matrix @ np.array([0, 0, end_point]),
            }
            colors = {'x': 'red', 'y': 'green', 'z': 'blue'}


            for axis_name, vec in axes.items():
                plot_data.append(go.Scatter3d(
                    x=[attractor_pos[end_point, 2], attractor_pos[end_point, 2] + vec[0]],
                    y=[attractor_pos[end_point, 0], attractor_pos[end_point, 0] + vec[1]],
                    z=[attractor_pos[end_point, 1], attractor_pos[end_point, 1] + vec[2]],
                    mode='lines',
                    line=dict(color=colors[axis_name], width=1),
                ))

        for goal_id in range(self.goals.shape[1]):
            r = R.from_quat([
                attractor_ori[goal_id, 1],
                attractor_ori[goal_id, 2],
                attractor_ori[goal_id, 3],
                attractor_ori[goal_id, 0],
                ])  # SciPy expects (x, y, z, w)

            rot_matrix = r.as_matrix()

            # Unit vectors
            axes = {
                'x': rot_matrix @ np.array([axis_len, 0, 0]),
                'y': rot_matrix @ np.array([0, axis_len, 0]),
                'z': rot_matrix @ np.array([0, 0, axis_len]),
            }
            colors = {'x': 'red', 'y': 'green', 'z': 'blue'}


            for axis_name, vec in axes.items():
                plot_data.append(go.Scatter3d(
                    x=[attractor_pos[goal_id, 2], attractor_pos[goal_id, 2] + vec[0]],
                    y=[attractor_pos[goal_id, 0], attractor_pos[goal_id, 0] + vec[1]],
                    z=[attractor_pos[goal_id, 1], attractor_pos[goal_id, 1] + vec[2]],
                    mode='lines',
                    line=dict(color=colors[axis_name], width=5),
                ))


        marker_data_attractors_demo = go.Scatter3d(
            x=attractors_demo[:, 2],
            y=attractors_demo[:, 0],
            z=attractors_demo[:, 1],
            marker=go.scatter3d.Marker(size=4, color='green'),
            opacity=0.8,
            mode='markers'
        )
        plot_data.append(marker_data_attractors_demo)
        #plot_data.append(marker_data_attractors)

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
