<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Learning Multi-Stable Motion Primitives</title>
</head>
<body>

<h1>Learning Multi-Stable Motion Primitives from Demonstrations via Multi-Attractor Dynamics</h1>

<p>
  This project explores the integration of <strong>dynamical systems</strong> and <strong>neural networks</strong> for learning motion primitives from demonstrations.
  Specifically, we investigate how to transfer the <strong>stability properties</strong> of known dynamical systems to unknown systems inferred from human demonstrations. 
  This falls under the domain of <strong>Imitation Learning</strong>.
</p>

<p>
  While several frameworks combine neural models with stable dynamics, few explicitly address <strong>multi-stability</strong> — the ability to encode and reproduce 
  multiple trajectories with distinct end-points. Our approach introduces and evaluates architectures capable of learning such multi-stable behaviors.
</p>

<p>
We integrate <strong>multi-attractor dynamics</strong> into the <strong>latent space</strong> of a neural network. 
After observing a few demonstrations, the model learns to reach target configurations from arbitrary initial states. Using a multivariate gaussian dynamic we are able to learn a potential infinite number of attractor in R^n, 
while  using a continuous set of attraction point we can constrain the final configuration to reach a continuous curve of attraction, allowing a robot to select any grasping point on a particular object.
Finally, with a multivariate gaussian dynamic on a spherical manifold, we are able to control also the orientation of the robot.

Encoding different attractors in memory enhances model flexibility and allows shaping of the underlying <strong>vector field</strong> used for planning. 🎯
</p>


<img src="./img/Squid-discrete-Arc2.png" alt="Discrete Attractors Architecture" style="width: 100%;">
<img src="./img/Squid-continuous-Arc2.png" alt="Continuous Attractors Architecture" style="width: 100%;">


<p>
<h2>🚀 Installation</h2>

<p>This project uses <strong>Poetry</strong> for dependency and environment management.</p>

<pre><code>git clone &lt;https://github.com/gg-dema/SQUID.git&gt; your_repo &amp;&amp; cd your_repo
poetry install
poetry shell
</code></pre>

<p>
  Make sure you have Poetry installed beforehand. You can find installation instructions at 
  <a href="https://python-poetry.org/docs/" target="_blank">python-poetry.org</a>.
</p>

<h2>🧠 Training the Model</h2>

<p>All model variants use the same entry point:</p>

<pre><code>cd src
python train.py --params &lt;config_file&gt;
</code></pre>

<p>Configuration files are located in <code>src/params/</code> and define the type of task:</p>
<ul>
  <li><strong>Discrete attractor points</strong>: Use files like <code>Discrete_Squid_{SPACE}{DIM}_1o.py</code>. Space could be <code>S</code> for spherical manifold or <code>R</code> for use euclidean metrics.</li>
  <li><strong>Continuous attractor curves</strong>: Use files prefixed with <code>shape_SHAPENAME.py</code></li>
</ul>

<h3> Dataset Setup</h3>

<ul>
  <li>For <strong>discrete attractors</strong>, you only need a dataset of demonstrations (see examples under <code>multi_attractors/</code> and <code>kuka/</code>).</li>
  <li>For <strong>continuous attractors</strong>, two inputs are required:
    <ol>
      <li>The main contour/curve of the attraction region</li>
      <li>A set of <strong>hard negatives</strong> (non-attractor points)</li>
    </ol>
    You can generate the hard negative set using:
<pre><code>python src/tool/hard_neg_extraction.py</code></pre>
  </li>
</ul>

<h2> Experiments</h2>
<h3> Reference dynamics</h3>

<p>
The reference dynamics used in the project are the multivariate gaussian dynamic in R^n and in S^n, a "continuous curve" dynamic. All of these are define in the <code>src/agent/neural_networks.py</code> file. 
In lower dimension, here you can have a look at the vector field generated from them: 
<img src="img/full_latent_model.png">
</p>

<h3>Planar motion in R^2: Discrete Target</h3>
<img src="img/3-r.jpg" style="width: 32%;">
<img src="img/3R-cyc.jpg" style="width: 32%;">
<img src="img/10-r2.jpg" style="width: 33%;">


<h3>Planar motion in R^2: Continuous Target</h3>
<p>2D vector fields generated usign a continuous cruve of attractions</p>
<img src="img/star.jpg" style="width: 24%;">
<img src="img/squirtle.jpg"  style="width: 24.5%;">
<img src="img/charmender.jpg"  style="width: 24%;">
<img src="img/bulbasaur.jpg"  style="width: 24%;">

<h3>Planar motion in S^2:</h2>
<p> 2D vector fields generated using a spherical latent space</p>
<img src="img/3-s.jpg" style="width: 48%">
<img src="img/10-s2.jpg" style="width: 48%">

<h3>Motion in SE(3):</h3>
<p>7-dim motion, where the state <code>q</code> is composed by 3 element for the position (<code>x, y, z</code>) and 4 element for the orientation (quaternion) (<code>alpha, beta, gamma, w</code>)</p>
<img src="img/7-dim-motion.png">


<h2>🙏 Acknowledgments</h2>

<p>
  This project builds upon the foundational work in the following repositories:
  <ul>
    <li><a href="https://github.com/rperezdattari/Stable-Motion-Primitives-via-Imitation-and-Contrastive-Learning" target="_blank">
      Stable Motion Primitives via Imitation and Contrastive Learning</a></li>
    <li><a href="https://github.com/rperezdattari/PUMA-Deep-Metric-IL-for-Stable-Motion-Primitives" target="_blank">
      PUMA: Deep Metric IL for Stable Motion Primitives</a></li>
  </ul>
  We thank the original authors for making their work publicly available.
</p>

</body>
</html>
