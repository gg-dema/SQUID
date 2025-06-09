import pickle
import plotly.graph_objects as go
import sys
# iteration = sys.argv[1]
path = '/home/dema/Project/Multi-Condor/multi_condor_2/src/results/1st_order_3D_box/0/images/primitive_0_iter_6000.pickle'
plot_data = pickle.load(open(path, 'rb'))
fig = go.Figure(data=plot_data['3D_plot'])
fig.show()
