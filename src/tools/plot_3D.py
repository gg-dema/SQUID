import pickle
import plotly.graph_objects as go
import sys
# iteration = sys.argv[1]
file = 'primitive_0_iter_32000.pickle'
folder = "/home/dema/Project/squid/src/results/franka-mariano/copy_kuka_small_ws_no_boundary/5/images"
path = folder + "/" + file
plot_data = pickle.load(open(path, 'rb'))
fig = go.Figure(data=plot_data['3D_plot'])
fig.show()


