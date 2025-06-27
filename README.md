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
  Specifically, we investigate how to transfer the <strong>stability properties</strong> of known dynamical systems to <em>unknown systems</em> inferred from human demonstrations. 
  This falls under the domain of <strong>Imitation Learning</strong>.
</p>

<p>
  While several frameworks combine neural models with stable dynamics, few explicitly address <strong>multi-stability</strong> — the ability to encode and reproduce 
  multiple trajectories with distinct end-points. Our approach introduces and evaluates architectures capable of learning such multi-stable behaviors.
</p>


<h2>Research Questions</h2>

<h3>✅ How can we address multiple demonstrations with a single model and enforce stability?</h3>

<p>
  We integrate <strong>multi-attractor dynamics</strong> into the <strong>latent space</strong> of a neural network. 
  After observing a variety of demonstrations, the model learns to reach target configurations from arbitrary initial states. 
  Encoding different attractors in memory enhances model flexibility and allows shaping of the underlying <strong>vector field</strong> used for planning.
</p>

<img src="./img/Squid-discrete-Arc2.png" alt="Discrete Attractors Architecture" style="width: 80%;">

<h3>🎯 How can we generalize attraction from a single point to an entire object contour?</h3>

<p>
  We extend attraction modeling from discrete points to <strong>continuous attractor regions</strong>, 
  such as the full contour of an object. This enables the planner to consider any point along the contour as a valid target.
</p>

<img src="./img/Squid-continuous-Arc2.png" alt="Continuous Attractors Architecture" style="width: 80%;">


<h3> with this project, we provide solution to the following cases:</h3> 
<p>
  We provide solutions for learning:
  <ul>
    <li>Discrete attractor points in ℝ^n</li>
    <li>Continuous attractor curves</li>
    <li>Attractors on manifold spaces, such as S^n</li>
  </ul>
</p>

<h2>🚀 Installation</h2>

<p>This project uses <strong>Poetry</strong> for dependency and environment management.</p>

<pre><code>git clone &lt;your_repo_link&gt; &amp;&amp; cd &lt;your_repo&gt;
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
  <li><strong>Discrete attractor points</strong>: Use files like <code>Discrete_Condor.py</code></li>
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
<pre><code>python src/tool/segment_image.py</code></pre>
  </li>
</ul>

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
