# LASA Dataset
LASA = ['Angle',
        'BendedLine',
        'CShape',
        'DoubleBendedLine',
        'GShape',
        'heee',
        'JShape',
        'JShape_2',
        'Khamesh',
        'Leaf_1',
        'Leaf_2',
        'Line',
        'LShape',
        'Multi_Models_1',
        'Multi_Models_2',
        'Multi_Models_3',
        'Multi_Models_4',
        'NShape',
        'PShape',
        'RShape',
        'Saeghe',
        'Sharpc',
        'Sine',
        'Snake',
        'Spoon',
        'Sshape',
        'Trapezoid',
        'Worm',
        'WShape',
        'Zshape']

# LAIR dataset
LAIR = ['e',
        'double_loop',
        'Lag',
        'double_lag',
        'phi',
        'mountain',
        'two_roads',
        'G_angle',
        'capricorn',
        'triple_loop',
        'two']

# Interpolation dataset
interpolation = ['interpolation_1',
                 'interpolation_2',
                 'interpolation_3']

# Optitrack dataset
optitrack = ['hammer']

# Joint space dataset
joint_space = ['cleaning_1',
               'cleaning_2',
               'picking',
               'picking_clean',
               'picking_mix']

multi_attractors = ['v0',  # 1 single attractor, done for testing if the dataset was properly loaded
                   'v1',  # 2 attractors
                   'v2',  # 3 attractors
                   'v3',  # 4 easy
                   'v4',  # 4 complex
                   'v5',  # 3 attractors mix
                   'v6',  # 3 attractors, triangle
                   'v7',  # 3 attractors, triangle, but pointing out
                   'v8',  # 10 attractors random
                   'v9',  # 5 attractors, star, pointing out
                   'v10',  # inf attractors, star, pointing inside
                   'v11',
                   'v12',
                   'v13', # crossed trajectory
                   'v14', # crossed and circle
                   'v15'
        ]


eval_dataset = [
    'v1',  # 2 attractors
    'v2',  # 2 attractors
    'v3',  # 2
    'v4',  # 3
    'v5',  # 3 attractors
    'v6',  # 3 attractors
    'v7',  # 3

]
kuka = ['box',
        'box_full',
        'simple',
        'split']

cycle = [
    'limit_cycle',
    'strange_cycle',
    'otto'
]
