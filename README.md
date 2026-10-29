
# Introduction to Deep Learning Phenomena

This repository contains lecture notes and related resources on deep learning, using observed phenomena as a starting point for understanding its core concepts. It is intended for researchers, practitioners, students, and others interested in deep learning.

## Course Videos

The complete course videos by Zhiqin Xu are available on [Bilibili](https://space.bilibili.com/95975441/channel/seriesdetail?sid=3749682), covering both introductory and advanced topics in deep learning.

## Updates (June 28, 2025)
1 Mainly corrected wording and expression errors.
2 Added Chapter 15, "Solving Differential Equations with Neural Networks."


## Updates (November 13, 2024)

### 1. Experiments
   - **New materials:** Added an experimental section with detailed lab notes and corresponding Jupyter notebooks. These materials provide practical deep learning examples and data analyses to reinforce the concepts in the lecture notes.

### 2. The Effect of Initialization on Reasoning
   - **Theory-driven applications:** Added a section discussing how initialization affects reasoning performance. It applies condensation theory to large language models, illustrating how theoretical insights can inform practical applications and connecting theory with practice.

### 3. Revisions to the Chapter on Models
   - **Variational autoencoder updates:** Improved the section on variational autoencoders.

## Chapter 1: Introduction to Deep Learning

### **Overview**

This chapter explains how neural networks work (backpropagation and gradient descent), the evolution of optimization algorithms, and the geometry of loss landscapes. It also explores the tension between deep learning and classical statistical learning theory: the **generalization puzzle**, in which overparameterized models can achieve better generalization. The chapter introduces **implicit bias** as a central concept for explaining this phenomenon and concludes by proposing a **phenomenon-driven** research paradigm.

---

### **Outline**

<details>
<summary><b>1. The Essence of Deep Learning</b></summary>

*   **Data Fitting:**
    *   Mathematically, deep learning seeks a function model that approximates the underlying patterns in data (the target function).
    *   The goal is to learn a mapping from the input space to the output space that performs well on unseen test data (generalization).
*   **Neural Network Architectures:**
    *   **How neurons work:** Inspired by biological neurons, artificial neurons process information through a weighted linear combination ($w^Tx + b$) and a nonlinear activation function (ReLU, sigmoid, tanh, etc.).
    *   **Multilayer structure:** A single-layer perceptron can solve only linearly separable problems (and therefore cannot solve XOR). By stacking nonlinear transformations, multilayer neural networks gain powerful expressive capabilities (the universal approximation theorem).
</details>

<details>
<summary><b>2. Model Training and Optimization</b></summary>

*   **Loss Functions:**
    *   Measure the discrepancy between model predictions and ground-truth labels.
    *   **Regression:** Common choices include mean squared error (MSE) and mean absolute error (MAE).
    *   **Classification:** Cross-entropy is commonly used, often with softmax to convert outputs into a probability distribution.
*   **Loss Landscapes:**
    *   Describe the geometry of the loss function as model parameters vary.
    *   Deep learning loss landscapes are typically nonconvex, high-dimensional surfaces with many local minima and saddle points.
*   **Optimization Methods:**
    *   **Backpropagation:** Uses the chain rule to compute gradients efficiently and determine parameter update directions.
    *   **Gradient descent (GD) and its variants:**
        *   **Stochastic gradient descent (SGD):** Uses a small number of samples at each step, reducing computation and introducing stochastic noise that can help escape local optima.
        *   **Momentum methods (Momentum/Nesterov):** Introduce "inertia" to traverse flat regions or saddle points.
        *   **Adaptive methods (AdaGrad, Adam):** Automatically adjust learning rates for individual parameters to accelerate convergence.
</details>

<details>
<summary><b>3. Key Training Details and Theoretical Limitations</b></summary>

*   **Parameter Initialization:**
    *   The choice of initial values determines the starting point and dynamical trajectory of training.
    *   Common methods (LeCun, Xavier, and Kaiming initialization) aim to stabilize signal variance as signals propagate through deep networks, preventing vanishing or exploding gradients.
*   **The No Free Lunch Theorem:**
    *   No single algorithm is optimal across all possible problem distributions.
    *   Deep learning succeeds because its architectures (such as CNNs and Transformers) match the specific structures of real-world data (such as images and text).
</details>

<details>
<summary><b>4. Core Theory: The Generalization Puzzle and Implicit Bias</b></summary>

*   **The Generalization Puzzle:**
    *   **Classical view:** Statistical learning theory suggests that more complex models (with more parameters) are more prone to overfitting (a U-shaped error curve).
    *   **Deep learning phenomenon:** Modern neural networks are **overparameterized** (with far more parameters than training samples), yet they can both fit the training data perfectly (zero training error) and generalize extremely well.
*   **Implicit Bias:**
    *   **Question:** Why do neural networks tend to select "good" solutions even without explicit regularization terms?
    *   **Explanation:** Particular neural network architectures, combined with particular optimization algorithms (such as SGD), have an inherent "implicit regularization" effect that favors learning low-complexity patterns first.
*   **Research Paradigm: Phenomenon-Driven Research**
    *   Given that mathematical theory lags behind engineering practice, this approach advocates first identifying macroscopic patterns (such as scaling laws) through extensive experiments, then building mathematical models to explain them.
</details>

## Chapter 2: Data and Neural Network Architectures



### **Overview**
This chapter first introduces **fully connected networks**, followed by **residual networks (ResNets)**, explaining how skip connections address the challenges of training deep networks. It then traces the evolution of architectures for different data types: **convolutional neural networks (CNNs)** were developed to accommodate the spatial locality and invariances of **image data**, while the need to capture long-range dependencies in **language data** and support parallel computation drove the transition from **recurrent neural networks (RNNs)** to the **Transformer** architecture, which transformed the AI paradigm.

---

### **Outline**

<details>
<summary><b>1. Fully Connected and Residual Network Architectures</b></summary>

*   **Fully Connected Networks (FCNs):**
    *   **Definition:** The most basic architecture, in which every neuron in one layer connects to every neuron in the next.
    *   **Limitations:** Although theoretically capable of universal approximation, these networks ignore the spatial and temporal structure of data. Their parameter counts grow dramatically for high-dimensional inputs (such as images), and their generalization is relatively weak.
*   **Residual Networks (ResNets):**
    *   **Core problem addressed:** "Degradation," in which training error increases as more layers are added, rather than an increase in overfitting.
    *   **Core mechanism:** Introduce **skip connections**.
    *   **Significance:** Effectively alleviate vanishing gradients in deep networks, making it possible to train networks with hundreds or thousands of layers.
</details>

<details>
<summary><b>2. Vision Tasks: From Biological Inspiration to Convolutional Networks (CNNs)</b></summary>

*   **Properties of Image Data:**
    *   **Local correlations:** Neighboring pixels are usually strongly correlated.
    *   **Invariances:** Translation, rotation, scale, and statistical invariance.
*   **Biological Inspiration:**
    *   Research on the visual cortex (area V1) identified simple cells (which detect edge orientations) and complex cells (which exhibit position invariance).
*   **Core CNN Components:**
    *   **Convolutional layers:** Use **local connectivity** and **weight sharing** to greatly reduce parameter counts and efficiently extract local features (such as edges and textures).
    *   **Pooling layers:** Perform downsampling (max pooling or average pooling), introducing translation invariance and enlarging the receptive field.
</details>

<details>
<summary><b>3. Language Tasks: From Recurrent Networks (RNNs) to Transformers</b></summary>

*   **Properties of Language Data:** Sequential structure, variable length, strong contextual dependencies, and discrete symbols.
*   **Recurrent Neural Networks (RNNs) and LSTMs:**
    *   **RNNs:** Carry memory through hidden states, but suffer from severe vanishing or exploding gradients and cannot parallelize computation across sequence steps.
    *   **LSTMs/GRUs:** Introduce gating mechanisms (such as forget and input gates) and memory mechanisms (such as the cell state) to address gradient problems in training on long sequences and capture long-range dependencies.
*   **Transformers (Key Focus):**
    *   **Major breakthrough:** Replace recurrence with an architecture based entirely on **attention**, enabling **parallel computation**.
    *   **Key techniques:**
        *   **Positional encoding:** Methods such as sinusoidal encoding and RoPE supply the sequence-order information missing from attention mechanisms.
        *   **Self-attention:** Query-key-value computations allow each word to attend to all words in the sequence and capture global dependencies.
        *   **Multi-head attention:** Extracts features in parallel across different subspaces, increasing expressive power.
    *   **Architectural variants:** Encoder-decoder (the original architecture), decoder-only (the GPT family, suited to generation), and encoder-only (BERT, suited to understanding).
</details>

<details>
<summary><b>4. An Introduction to Generative Models</b></summary>

*   **Autoencoders (AEs):**
    *   Consist of an encoder (which compresses data) and a decoder (which reconstructs data), and learn low-dimensional representations.
*   **Variational Autoencoders (VAEs):**
    *   Extend autoencoders by introducing probability distributions (typically Gaussian).
    *   Use the **reparameterization trick** to make the sampling process differentiable.
    *   **Characteristics:** A continuous latent space enables the generation of new samples, rather than merely memorizing training data.
</details>

## Chapter 3: The Curse of Dimensionality

### **Overview**

This chapter examines a central challenge in machine learning and data science: the **curse of dimensionality**. It first reveals counterintuitive geometric properties of high-dimensional spaces, including extreme data sparsity, the concentration of volume near the surface (in a thin spherical shell), and the tendency of random vectors to be orthogonal. Examples from numerical integration, partial differential equations, and function approximation then demonstrate why traditional grid-based algorithms fail in high dimensions. Finally, the chapter discusses strategies for overcoming the curse of dimensionality, from classical **Monte Carlo methods**, which use randomness to avoid dimension dependence, to modern **neural network methods**. In particular, **Barron space** theory provides a mathematical explanation for the success of deep learning on high-dimensional data (such as images and text), establishing dimension-independent approximation and generalization error rates in specific function spaces.

---

### **Outline**

<details>
<summary><b>1. Properties of Data in High-Dimensional Spaces</b></summary>

*   **Data Sparsity:**
    *   In a high-dimensional unit cube, the number of samples needed to cover all subregions grows exponentially with dimension ($n \ge k^d$).
    *   Real-world high-dimensional data (such as images) are extremely sparse in the ambient space, leaving most regions "empty."
*   **Volume Concentration Near the Surface:**
    *   As dimension increases, nearly all the volume of a unit ball lies in a very thin spherical shell near its surface.
    *   **Experimental evidence:** When a neural network is trained to fit a high-dimensional quadratic function, errors near the boundary are very small, while errors near the center (in the interior) are large because almost no samples lie there.
*   **Distance Concentration and Orthogonality:**
    *   **Distance concentration:** Distances between randomly sampled points approach a dimension-dependent constant (such as $\sqrt{2}$) with very small variance, undermining the notion of a "nearest neighbor."
    *   **Angle concentration:** The angle between random vectors is very likely to be close to 90 degrees (orthogonal), making cosine similarity ineffective in some settings.
*   **The Gaussian Annulus Theorem:**
    *   Samples from a high-dimensional Gaussian distribution do not cluster at the origin, where the probability density is highest. Instead, they lie in an annulus with a radius of approximately $\sqrt{d}$.
*   **Dimensionality Reduction by Random Projection:**
    *   **The Johnson-Lindenstrauss lemma:** High-dimensional data can be projected into a lower-dimensional space through a random linear mapping while approximately preserving pairwise distances.
*   **Linear Separability of Data:**
    *   High-dimensional spaces offer more degrees of freedom, allowing data that are not linearly separable in low dimensions (such as XOR) to become linearly separable in higher dimensions (the principle behind kernel methods).
</details>

<details>
<summary><b>2. Examples of the Curse of Dimensionality</b></summary>

*   **High-Dimensional Numerical Integration:**
    *   Traditional grid-based methods (such as the trapezoidal rule) require a number of nodes that grows exponentially with dimension ($n^d$), making computation infeasible.
*   **High-Dimensional Partial Differential Equations (PDEs):**
    *   Examples include the Schrödinger equation in quantum physics ($3d$ dimensions) and the Black-Scholes equation in financial engineering (whose dimension depends on the number of assets). Traditional finite-difference methods face an explosion in the number of degrees of freedom.
*   **High-Dimensional Function Approximation:**
    *   The approximation error is $\epsilon \approx n^{-s/d}$. The dimension $d$ in the denominator greatly slows convergence, making approximation difficult even for very smooth functions.
</details>

<details>
<summary><b>3. Overcoming the Curse of Dimensionality: Monte Carlo Methods</b></summary>

*   **Monte Carlo Methods:**
    *   **Core idea:** Use random sampling and statistical averaging to approximate quantities of interest.
    *   **Advantage:** The convergence rate is $O(n^{-1/2})$, **independent of the dimension $d$**.
    *   **Applications:** Because volume concentrates near the surface, Monte Carlo methods naturally sample these regions. AlphaGo uses Monte Carlo tree search to address search in the high-dimensional state space of Go.
    *   **Limitation:** Convergence is relatively slow, so many samples are needed to substantially reduce error.
</details>

<details>
<summary><b>4. Overcoming the Curse of Dimensionality: Neural Networks and Barron Spaces</b></summary>

*   **Limitations of the Universal Approximation Theorem:**
    *   Although it establishes that neural networks can approximate arbitrary continuous functions, it does not specify how the required number of parameters depends on dimension.
*   **Barron Space Theory:**
    *   **Definition:** A class of function spaces with specific integral representations, developed for analyzing neural networks.
    *   **Main results (E et al., 2019):**
        *   **Approximation:** A two-layer neural network of width $m$ approximates functions in Barron space with an error of $O(m^{-1/2})$, **independent of the dimension $d$**.
        *   **Generalization:** The generalization error converges at a rate of $O(n^{-1/4})$, also **independent of the dimension $d$**.
    *   **Significance:** Provides a theoretical explanation for why neural networks can avoid the curse of dimensionality when handling high-dimensional data.
</details>

## Chapter 4: The Frequency Principle

### **Overview**

This chapter introduces the **Frequency Principle** in neural network training: networks tend to **learn low frequencies first and high frequencies later**. This implicit bias explains why overparameterized neural networks often find smooth solutions that generalize well when fitting data, rather than highly oscillatory solutions that fit noise. Starting with experiments on simple one-dimensional functions and progressing to high-dimensional image tasks, the chapter uses mathematical tools such as Fourier analysis to reveal differences in how quickly neural networks learn different frequency components. It also examines **early stopping** through the lens of the Frequency Principle and identifies limitations of neural networks when handling high-frequency information, such as in high-dimensional numerical simulations.

---

### **Outline**

<details>
<summary><b>1. Observations and Definition of the Frequency Principle</b></summary>

*   **Observations in Low-Dimensional Experiments:**
    *   When fitting simple one-dimensional functions such as $\sin(x)$, neural networks tend to connect data points with **smooth** curves rather than highly oscillatory ones.
    *   Even when the target function contains high-frequency components, such as $\sin(5x)$, the network first fits low-frequency components, such as $\sin(x)$, learning the overall shape before finer details and textures.
*   **Definition of the Frequency Principle:**
    *   During training, neural networks fit the target function in order **from low to high frequencies**.
    *   Low-frequency components converge quickly, while high-frequency components converge slowly.
    *   This phenomenon is widely observed across different activation functions, loss functions, and network architectures.
</details>

<details>
<summary><b>2. Theoretical and Practical Implications of the Frequency Principle</b></summary>

*   **Explaining Generalization:**
    *   Networks learn low-frequency (smooth) information first, and real-world data are typically dominated by low frequencies, while noise often lies at high frequencies. This bias allows networks to learn useful patterns early in training, providing inherent noise resistance and generalization ability.
*   **The Mechanism of Early Stopping:**
    *   From a frequency perspective, early stopping ends training after the network has learned the low-frequency signal but before it learns high-frequency noise, acting as a **low-pass filter** to prevent overfitting.
*   **Limitations of Neural Networks:**
    *   **The High-Frequency Challenge:** Standard neural networks are difficult to train and converge extremely slowly on tasks **dominated by high-frequency components**, such as solving certain partial differential equations, reconstructing highly detailed images, or classifying parity functions.
    *   This explains why networks perform like random guessing when fitting parity functions, which exhibit high-frequency oscillations.
</details>

<details>
<summary><b>3. Methods for Frequency Analysis in High-Dimensional Spaces</b></summary>

*   **Challenges:**
    *   Fourier transforms of high-dimensional data, such as images, face an explosion in computational cost (the curse of dimensionality), making direct analysis difficult.
*   **Tools for High-Dimensional Analysis:**
    *   **Projection Method:** Project high-dimensional data onto low-dimensional directions, such as principal component directions, and then apply one-dimensional Fourier analysis.
    *   **Filtering Method:** Use Gaussian filtering to separate signals into low- and high-frequency components, then track the convergence of their respective relative errors during training.
</details>

<details>
<summary><b>4. Results of High-Dimensional Experiments</b></summary>

*   **Image Classification (CIFAR10):**
    *   Experiments confirm that neural networks still follow the Frequency Principle on real-world high-dimensional data: low-frequency components converge first, followed by high-frequency components.
</details>

## Chapter 5: Designing Efficient Neural Networks Based on the Frequency Principle

### **Overview**


This chapter introduces several improvement strategies based on the Frequency Principle and demonstrates their successful applications in scientific computing, such as solving partial differential equations, and computer vision, such as neural radiance fields (NeRF).

---

### **Outline**


<details>
<summary><b>Designing Efficient Neural Networks Based on the Frequency Principle</b></summary>

*   **Multiscale Neural Networks:**
    *   **Core Idea:** Rescale input coordinates at different scales ($k \cdot x$) so that originally high-frequency components appear as low frequencies to the network, accelerating convergence.
    *   **Architecture:** Use multiple subnetworks, each responsible for a different frequency range, in a manner analogous to Fourier series decomposition.
    *   **Applications:** These methods have successfully addressed problems that are difficult for conventional networks, including high-frequency wave propagation and multiscale partial differential equations.
*   **Neural Networks Based on Subspace Decomposition:**
    *   Explicitly decompose the target function into low- and high-frequency components, fit them with separate modules, and introduce orthogonality constraints to further improve accuracy.
*   **Neural Radiance Fields:**
    *   Without positional encoding (a particular form of Fourier features), NeRF can render only blurry, low-frequency images; adding it enables the capture of sharp, high-frequency textures and edges.
*   **Fourier Feature Networks:**
    *   **Core Idea:** Introduce sine and cosine mappings with random frequencies at the input layer: $\gamma(v) = [\cos(2\pi b v), \sin(2\pi b v)]$.
</details>

## Chapter 6: Mechanisms Underlying the Frequency Principle

### **Overview**

This chapter examines the mathematical mechanisms and factors underlying the **Frequency Principle**. Using the **Linear Frequency Principle (LFP) model** and an **equivalent variational problem**, it theoretically quantifies how quickly neural networks converge at different frequencies. It analyzes the effects of **initial weight magnitude**, **activation function type**, and **loss function form**, showing that excessively large initial weights can invalidate the Frequency Principle, while certain loss functions, such as those containing derivative terms, can accelerate high-frequency convergence. It also introduces the **FP-norm** and **FP-space** to establish a theory of a priori generalization error estimates.

---

### **Outline**

<details>
<summary><b>1. Factors Affecting the Frequency Principle</b></summary>

*   **Initial Weight Magnitude:**
    *   **Observation:** Large initial weights weaken or even disrupt the Frequency Principle.
    *   **Explanation:** Large weights cause the initial output to contain more high-frequency components (strong oscillations), accelerating high-frequency fitting early in training so that learning no longer strictly proceeds from low to high frequencies.
    *   **Consequence:** Small initialization promotes smooth learning, whereas large initialization can introduce high-frequency interference and impair generalization.
*   **Effects of Activation Functions:**
    *   **Monotonically Decaying Class (e.g., tanh, ReLU):** Their Fourier transforms decay monotonically with frequency, making the Frequency Principle readily observable.
    *   **Nonmonotonic Class (e.g., Ricker):** If an activation function is not monotonically decreasing in the frequency domain, for example, if it first rises and then falls, the convergence order of the corresponding frequencies changes, causing the Frequency Principle to fail.
*   **Effects of the Loss Function:**
    *   **Standard Mean Squared Error (MSE):** Follows the standard Frequency Principle.
    *   **Losses Involving Derivatives (e.g., Sobolev Norms):** Differentiation corresponds to multiplication by a frequency factor $k$ in the frequency domain, effectively increasing the weight of high-frequency components and substantially accelerating their convergence.
</details>

<details>
<summary><b>2. A Simple Theoretical Analysis of the Frequency Principle</b></summary>

*   **Fourier Analysis Perspective:**
    *   The Fourier transform of a two-layer tanh network reveals an amplitude spectrum that decays exponentially with frequency ($e^{-|\pi k / 2w|}$).
    *   **Dynamical Equations:** The derived gradient descent dynamics show that gradient magnitudes for low-frequency components are much larger than those for high-frequency components, causing low frequencies to converge first.
*   **Key Conclusion:** Deep learning's preference for low frequencies arises from the **smoothness of activation functions** and the update mechanism of **gradient descent**.
</details>

<details>
<summary><b>3. The Linear Frequency Principle (LFP) Model</b></summary>

*   **Model Construction:**
    *   Assuming a wide network (many neurons) and a small learning rate, approximate neural network dynamics by a linear evolution equation.
    *   Derive how the neural network output $u$ evolves over time $t$, yielding a form similar to the heat equation in the frequency domain.
*   **Equivalent Variational Problem:**
    *   Neural network training is shown to be equivalent to solving a variational problem: $\min \int (\gamma(\xi))^{-2} |\hat{f}(\xi)|^2 d\xi$.
    *   **Physical Interpretation:** The network seeks a function whose high-frequency components are penalized by $\gamma(\xi)^{-2}$, which grows rapidly with frequency, forcing the solution to remain smooth and dominated by low frequencies.
    *   **Antisymmetric Initialization (ASI):** The proposed ASI technique subtracts two identically initialized networks to eliminate the effect of a nonzero initial output, ensuring that the model learns low frequencies starting exactly from zero.
</details>

<details>
<summary><b>4. Limits of High-Frequency Convergence and FP-Space</b></summary>

*   **FP-Norm and FP-Space:**
    *   A norm specific to neural networks (the FP-norm) is defined to measure a function's complexity from the network's perspective.
    *   **Generalization Error Bound:** The upper bound on generalization error is shown to scale with the number of training samples $n$ as $O(n^{-1/4})$. This convergence rate is **independent of the data dimension $d$**, further demonstrating the ability of neural networks to overcome the curse of dimensionality.
*   **Limits of High-Frequency Convergence:**
    *   **Critical Exponent $\alpha = d$:** In the variational model, the decay exponent $\alpha$ of the frequency penalty must exceed the data dimension $d$ for the solution to be smooth; if $\alpha < d$, the solution degenerates, taking nonzero values only at the training points and being zero elsewhere.
    *   **Implication:** The intrinsic frequency decay of neural networks makes fitting extremely high-frequency functions difficult, providing a mathematical explanation for their preference for smoothness.
</details>

## Chapter 7: Phase Diagram Analysis

### **Overview**

This chapter explores the decisive influence of **parameter initialization** on neural network training dynamics. Borrowing the concept of a phase diagram from thermodynamics, it defines state variables, such as the relative distance of parameter changes, to divide training dynamics into a **linear regime** and a **nonlinear regime**, also called the condensation regime. Through experimental observations and rigorous scaling analysis, the chapter constructs a phase diagram for two-layer fully connected ReLU networks, revealing deep connections among network width, initialization variance, and feature learning ability.

---

### **Outline**

<details>
<summary><b>1. Neural Network Behavior under Different Initializations</b></summary>

*   **Phenomenon-Driven Observations:**
    *   Experiments show that the scale of parameter initialization directly determines the shape of the function ultimately learned by the model.
    *   **Large Initialization:** The learned function exhibits jagged fluctuations.
    *   **Small Initialization:** The learned function exhibits linear interpolation.
*   **Distinguishing Linear and Nonlinear Behavior:**
    *   **Linear Behavior (Lazy Training):** When parameters change very little, the network can be approximated by a first-order Taylor expansion around initialization, similar to kernel methods or the NTK; the loss function is approximately convex in this setting.
    *   **Nonlinear Behavior (Feature Learning):** Parameters change substantially, preventing approximation by a linear model and allowing the network to adapt its internal representations to features in the data.
    *   **Metric:** **Relative Distance (RD)** measures how far the parameters at the end of training have moved relative to their initial values.
</details>

<details>
<summary><b>2. Separating the Linear and Nonlinear Regimes: Constructing the Phase Diagram</b></summary>

*   **Dynamical Phase Transition:**
    *   As the network width $m \to \infty$, RD exhibits bifurcation behavior: it either tends to $0$ (the linear regime) or diverges to infinity (the nonlinear regime).
*   **Components of the Phase Diagram:**
    *   **Linear Regime:** RD tends to 0. Parameters barely move, and the network behaves like a random feature model.
    *   **Condensed Regime:** RD tends to infinity. Parameters move substantially and exhibit **condensation**, in which many neuron weight vectors align with particular feature directions. This is central to deep learning's ability to go beyond traditional kernel methods.
    *   **Critical Regime:** The boundary between the linear and condensed regimes.
</details>

<details>
<summary><b>3. Condensation</b></summary>

*   **Definition:** During training in the nonlinear regime (small initialization), neuron weight vectors no longer remain randomly distributed but instead **condense** along a few specific directions.
*   **Significance:** Condensation is a hallmark of effective **feature learning** in neural networks.
</details>


## Chapter 8: Condensation

### **Overview**

This chapter examines **condensation**, a central phenomenon in the nonlinear training regime of neural networks. During training, neurons in the same layer tend to align their input weight directions and produce similar output functions. This mechanism reveals how neural networks improve generalization and feature learning efficiency by implicitly reducing model complexity, making a wide network functionally equivalent to a narrow one. The chapter demonstrates the prevalence of condensation through experiments with fully connected, convolutional, and residual networks. It also explains how condensation emerges at the start of training (**initial condensation**) and how **Dropout** can explicitly promote it.

---

### **Outline**

<details>
<summary><b>1. Experimental Observations of Condensation</b></summary>

*   **Description:** During training, the weight directions of initially random neurons gradually concentrate around a few specific directions, while their magnitudes increase, as if a few dominant directions were "growing" out of the network.
*   **Prevalence:**
    *   **Fully connected networks:** When fitting one-dimensional functions, neuron weights condense from a random distribution into a few discrete directions.
    *   **Convolutional neural networks (CNNs):** When training ResNet on CIFAR10, the weights of the final convolutional layer exhibit a clear block structure, indicating that different convolutional kernels become similar.
    *   **Residual networks (ResNet):** Condensation also occurs in deep networks and is closely related to the activation function, which determines the number of condensation directions.
*   **Conclusion:** Condensation is independent of network width, depth, architecture, and dataset size; it is a characteristic of nonlinear neural network training.
</details>

<details>
<summary><b>2. Defining and Understanding Condensation</b></summary>

*   **Definition:** During neural network training, neurons in the same layer exhibit a pronounced tendency to behave similarly.
    *   **Weight condensation:** Input weight vectors align in direction, with cosine similarity approaching 1 or -1.
    *   **Function condensation:** Neurons develop similar response functions to their inputs.
*   **Theoretical significance:**
    *   **Complexity control:** Condensation makes a wide network with many neurons functionally equivalent to a narrow network with only a few effective neurons, implicitly reducing the model's effective parameter count and preventing overfitting.
</details>

<details>
<summary><b>3. Initial Condensation</b></summary>

*   **Observation:** With small initialization, the loss stagnates early in training. During this period, neuron weights remain near the origin, but their directions rapidly adjust and align.
*   **Mechanism:**
    *   Analysis of the dynamical equations reveals specific stable lines in the gradient field near $0$.
    *   Driven by the gradient, neurons are rapidly attracted toward these stable lines.
    *   The number of condensation directions depends on the order (multiplicity) of the leading term in the activation function's Taylor expansion at zero; for example, tanh gives $2$ directions, while x*tanh gives $4$ directions.
</details>

<details>
<summary><b>4. How Dropout Promotes Condensation</b></summary>

*   **Experimental findings:** Even with large initialization, which typically leads to linear behavior, introducing Dropout can induce condensation and substantially smooth the model's output.
*   **Implicit regularization:**
    *   Deriving the equivalent loss function for Dropout reveals an explicit regularization term $R_1(\theta)$.
    *   This term penalizes differences between neuron outputs, forcing the weight directions of neurons in the same layer to align in parallel, thereby providing a mathematical explanation for how Dropout promotes condensation.
*   **Generalization benefits:** By promoting condensation, Dropout reduces the minimum number of samples needed to fit a model, enabling good generalization even with limited data.
</details>

## Chapter 9: The Embedding Principle of Loss Landscapes

### **Overview**

This chapter introduces the **embedding principle**, which reveals intrinsic connections between the loss landscapes of neural networks with different widths and depths. It begins with the experimental observation of **width similarity**: during training, networks of different widths stagnate at the same loss values and produce highly similar output functions. Building on this observation, the chapter mathematically defines how operations such as **neuron splitting** can "embed" critical points of a narrow network into the parameter space of a wider network.

---

### **Outline**

<details>
<summary><b>1. Observing Width Similarity</b></summary>

*   **Loss stagnation:**
    *   During training, neural networks of different widths, from narrow to wide, often stagnate at nearly identical loss values, producing staircase-shaped loss curves.
*   **Similarity of output functions:**
    *   At these stagnation points, the learned output functions are highly similar despite differences in the dimensionality of the networks' parameter spaces.
    *   **Internal representations:** Visualizations show that neurons inside a wide network condense at stagnation points, making its effective structure equivalent to that of a narrower network.
</details>

<details>
<summary><b>2. The Core Principle: The Embedding Principle</b></summary>

*   **Core definition:**
    *   The loss landscape of a wider neural network "contains" the critical points of all narrower networks.
*   **Mechanism: Neuron splitting**
    *   Splitting a neuron in a narrow network into two neurons with identical input weights and proportionally divided output weights embeds the narrow network into a wider one without changing its output function.
    *   This means that the parameter space of a wide network contains subspaces corresponding to solutions of narrower networks.
</details>

<details>
<summary><b>3. Why Are Wider Networks Easier to Train?</b></summary>

*   **Energy spectrum analysis of the loss landscape:**
    *   Statistics from extensive experiments reveal discrete "energy levels" in the distribution of loss values, corresponding to the losses of embedded "perfect" narrow networks of different widths.
*   **Hessian eigenvalue analysis (the key mechanism):**
    *   **Traps in narrow networks:** In a narrow network, a critical point may be a **local minimum**, with all Hessian eigenvalues positive, causing training to stagnate with no escape route.
    *   **Escape routes in wide networks:** After this point is embedded into a wider network, the Hessian acquires **negative eigenvalues**, indicating descent directions. The former "trap" becomes a **saddle point**.
    *   **Conclusion:** Wider networks have more potential descent directions, making it easier to escape suboptimal solutions and converge to global minima with better generalization.
</details>

<details>
<summary><b>4. Extensions: Depth and Critical Manifolds</b></summary>

*   **Critical manifolds:**
    *   Embedding is not unique: neurons can be split in multiple ways. A single critical point of a narrow network therefore corresponds to a high-dimensional "critical manifold" in a wider network.
    *   This manifold is highly degenerate and exerts an important attracting influence on optimization dynamics.
*   **The embedding principle for depth:**
    *   The theory applies to **depth** as well as width.
    *   **Layer linearization:** Inserting linear layers that approximate the identity map embeds solutions of shallow networks into deeper networks.
    *   The loss landscapes of deeper networks therefore also contain the critical points of shallower networks, explaining the distinct stages observed during deep network training.
</details>

## Chapter 10: Optimistic Estimate

### **Overview**

This chapter introduces the **optimistic estimate**. Under certain conditions, such as condensation, nonlinear models including neural networks can behave like simpler models. The chapter introduces **model rank** to quantify the minimum number of samples needed to recover a target function, referred to as the **optimistic sample size**. Theory and experiments show that, for neural networks, this sample size depends only on the target function's **intrinsic width** (complexity), rather than the actual width of the network.

---

### **Outline**

<details>
<summary><b>1. Model Rank and the Optimistic Estimate</b></summary>

*   **Core question:** Why can overparameterized models generalize well and even recover a target function when the number of samples is far smaller than the number of parameters?
*   **Definition: Model rank**
    *   Model rank describes the degrees of freedom of a model's local linearization near a given parameter point, namely the dimension of its tangent space.
    *   For linear models, model rank equals the number of parameters; for nonlinear models, it varies with the parameter location.
*   **Optimistic sample size:**
    *   The minimum number of samples needed to recover a target function when initialization is near the "optimal" parameter points (the target set).
    *   It provides a theoretical lower bound on the sample size required in actual training.
</details>

<details>
<summary><b>2. Theoretical Validation and Comparative Experiments</b></summary>

*   **Simple nonlinear regression:**
    *   Compare linear models with nonlinear models such as $f = \theta_0 + \theta_1x_1 + \theta_2\theta_3x_2$.
    *   Experiments show that an overparameterized nonlinear model can recover a specific target function from very few samples, matching the optimistic sample size, whereas a linear model cannot.
*   **Matrix factorization (matrix completion):**
    *   A nonlinear matrix factorization model ($W=AB$) can recover a low-rank matrix using far fewer samples than a linear model that directly optimizes $W$.
    *   Experiments confirm that the required sample size depends on the matrix rank, consistent with the theoretically derived formula for the optimistic sample size.
*   **Neural network models:**
    *   Experiments show that the sample size needed for neural networks to recover a target function is close to the theoretical optimistic sample size.
    *   **The role of hyperparameters:** Appropriate hyperparameters, such as small initialization, can guide a model into "good" regions where condensation occurs, bringing practical performance close to the optimistic estimate.
</details>

<details>
<summary><b>3. Implications for Neural Network Architecture Design</b></summary>

*   **The "free" expressive power of width:**
    *   **Conclusion:** Increasing network width, whether by adding neurons to fully connected layers or adding convolutional kernels, does not increase the optimistic sample size needed to recover a target function.
    *   **Significance:** Very wide networks can enhance expressive power without requiring an exponential increase in training data. The optimistic sample size depends only on the target function's **intrinsic width**.
*   **The "costly" expressive power of connections:**
    *   **Conclusion:** Adding connections between neurons, such as replacing a convolutional layer with a fully connected layer, substantially increases the optimistic sample size.
    *   **Comparison:** CNNs with shared weights require far fewer samples than networks without weight sharing and fully connected networks. This explains why CNNs outperform fully connected networks on image tasks, where locality is strong.
</details>


## Chapter 11: Flatness of Solutions

### **Overview**

This chapter introduces **flatness** and its relationship to model generalization. Empirical observations suggest that "flat" minima typically generalize better than "sharp" minima. The chapter discusses three factors that affect flatness: **batch size**, **the noise structure of stochastic gradient descent (SGD)**, and **Dropout**. It concludes with the **edge of stability** phenomenon observed during training, revealing the complex dynamics of gradient descent in nonconvex optimization.

---

### **Outline**

<details>
<summary><b>1. Flatness and Its Significance</b></summary>

*   **Definition:**
    *   Intuitively, flatness refers to the curvature of the loss function near a minimum.
    *   **Mathematical measures:** Flatness is commonly measured using Hessian eigenvalues, especially the largest eigenvalue $\lambda_{max}$. A smaller $\lambda_{max}$ indicates a flatter solution, while a larger $\lambda_{max}$ indicates a sharper one.
    *   Another measure considers how much the loss changes under random perturbations of the parameters.
*   **Relationship to generalization:**
    *   **Flat minima:** Insensitivity to parameter perturbations suggests that performance remains good on test data with a slightly shifted distribution, yielding strong generalization.
    *   **Sharp minima:** High sensitivity to parameter changes makes these solutions prone to overfitting and poor generalization.
</details>

<details>
<summary><b>2. Key Factors Affecting Flatness</b></summary>

*   **Batch size:**
    *   **Small batches:** Introduce greater stochastic noise, helping the model escape sharp local minima and favoring convergence to flat solutions with better generalization.
    *   **Large batches:** Provide more accurate gradient estimates but can become trapped in the nearest minimum, which may be sharp, reducing generalization performance.
*   **The noise structure of stochastic gradient descent (SGD):**
    *   **Anisotropic noise:** SGD noise is not uniform Gaussian noise; its covariance matrix aligns with the Hessian.
    *   **Escape mechanism:** Gradient noise is greater in directions of high curvature, forcing SGD to leave sharp regions quickly. Noise is smaller in flat regions, making it easier to remain there.
    *   **Implicit regularization:** SGD is equivalent to adding a penalty term to the loss function that penalizes gradient variance, thereby favoring flat solutions.
*   **Dropout:**
    *   **Effect:** Randomly dropping neurons subjects the model to strong perturbations.
    *   **Flattening effect:** To maintain a low loss under these perturbations, the model must find solutions that are insensitive to missing parameters, namely flat solutions. This explains how Dropout improves generalization.
</details>

<details>
<summary><b>3. The Edge of Stability</b></summary>

*   **Description:**
    *   When neural networks are trained with gradient descent (GD), the sharpness of the loss ($\lambda_{max}$) spontaneously increases and remains near $2/\eta$, where $\eta$ is the learning rate.
*   **Dynamical explanation:**
    *   **Stability threshold:** Linear stability theory indicates that optimization becomes unstable and oscillatory when $\lambda_{max} > 2/\eta$.
    *   **Self-regulation:** Training continually moves along the boundary between stability and instability. This challenges traditional convergence theory, suggesting that modern neural network training often involves a dynamic equilibrium at the edge of chaos rather than simple monotonic descent.
</details>

## Chapter 12: Anchor Functions: A Class of Simple Functions for Studying Language Models

### **Overview**

This chapter introduces the **anchor function** framework, developed to address challenges in large language model (LLM) research, including unspecified tasks, high computational costs, and limited interpretability. The framework defines a series of simplified **language-like tasks**, such as identity learning from "3x to x". These tasks retain the core "anchor-key" logical structure of natural language while removing the complications of natural linguistic context.

---

### **Outline**

<details>
<summary><b>1. Research Background and Approach</b></summary>

*   **Challenges:**
    *   **Unspecified tasks:** The next-token prediction training objective is not directly tied to specific downstream task objectives.
    *   **High costs:** Training large models requires vast amounts of data and computation, making results difficult to reproduce in academia.
    *   **Limited interpretability:** Reasoning processes are complex, making their underlying mechanisms difficult to identify.
*   **Characteristics of language tasks:**
    *   **Logical reasoning chains** and **anchor-key structures** are widespread in natural language.
</details>

<details>
<summary><b>2. Anchor Functions and Language-Like Tasks</b></summary>

*   **Anchor function:**
    *   Definition: A mapping from a sequence to a label, where the output is determined by the "anchor" in the sequence.
</details>

<details>
<summary><b>3. Data Splits and Generalization Evaluation</b></summary>

*   **Data-splitting strategies:** Special splits are designed to rigorously evaluate generalization.
    *   **Modulo-remainder split:** Ensures that anchor-key combinations in the training and test sets are completely disjoint.
    *   **Anchor-based split:** Certain anchors appear only in the training set, and their performance in new combinations is evaluated.
    *   **Composite-anchor split:** Trains on some combinations and tests the logic of unseen combinations.
*   **Types of generalization:**
    *   **Data generalization:** The task is familiar, but the data samples are unseen, such as performing addition on new numbers after learning the operation.
    *   **Task generalization:** The combination is unseen, such as testing the composition A+B after learning operations A and B separately; this is more challenging.
</details>

<details>
<summary><b>4. Shift and Broadcast</b></summary>

*   **Core mechanism:** Analysis of a simplified two-layer Transformer reveals that solving anchor-function tasks relies on two key operations:
    *   **Shift:** The first attention layer moves information about the key to the anchor position.
    *   **Broadcast:** The second attention layer broadcasts information from the anchor position to the output position.
*   **Evidence of broader applicability:**
    *   Similar attention-head patterns are observed in the full-scale **Llama2-7B** model: shallower layers tend to perform shifts, while deeper layers tend to perform broadcasts.
    *   This demonstrates that anchor functions provide an effective benchmark for understanding the internal workings of complex large models.
</details>

## Chapter 13: The Effect of Complexity Control on Reasoning in Language Models

### **Overview**

This chapter examines how hyperparameters affect large language models (LLMs) on complex reasoning tasks, especially compositional generalization. Its central claim is that **model hyperparameters, such as initialization scale and weight decay, determine whether a model tends toward rote memorization (memorization-based solutions) or the ability to apply learned rules to new situations (reasoning-based solutions)**. Composite-function tasks demonstrate the crucial role of **complexity control**: strong complexity control (small initialization and large weight decay) guides models toward structured, generalizable reasoning rules, leading to strong performance on out-of-distribution (OOD) tasks. Weak complexity control, by contrast, encourages memorization of training data and poor generalization.

---

### **Outline**

<details>
<summary><b>1. Introduction and Background: The "Genes" of Model Capabilities</b></summary>

*   **Research motivation:** Does a model truly learn reasoning rules, or does it merely memorize input-output mappings?
*   **Central metaphor:** Just as genes determine biological traits, such as skin and eye color, a model's hyperparameters, including initialization and weight decay, determine its level of reasoning ability: recording data or discovering rules.
</details>

<details>
<summary><b>2. Experimental Framework: Composite Functions</b></summary>

*   **Task definition:** Composite-function tasks take the form $f(x) = g(g(x; a_i); a_j)$.
    *   **Anchors:** $a_i, a_j$ represent specific operations, such as +1 or -2.
    *   **Key token:** $x$ is the number on which the operations are performed.
*   **Data splits:**
    *   **Training set / ID test set:** Contains a subset of anchor combinations, such as $a, b; a, c$.
    *   **OOD test set:** Contains anchor combinations unseen during training, such as $c, d$, to evaluate compositional generalization.
*   **Key variables:**
    *   **Initialization rate ($\gamma$):** Controls the magnitude of parameter initialization. A larger $\gamma$ corresponds to a smaller initialization scale.
    *   **Weight decay:** Controls the strength of $L_2$ regularization.
</details>

<details>
<summary><b>3. How Complexity Control Affects Learning</b></summary>

*   **Three learning regimes:**
    *   **Regime 1 (poor performance on both):** With extremely large or extremely small initialization, the model fails to generalize to either ID or OOD data.
    *   **Regime 2 (memorization-based solutions):** With moderate initialization, the model performs well on ID data but poorly on OOD data. It memorizes each composite anchor pair, such as $c, d$, as a separate label without understanding the rules for individual anchors, treating $c,d$ and $d,c$ differently.
    *   **Regime 3 (reasoning-based solutions):** **Small initialization + large weight decay** yields strong performance on both ID and OOD data. The model learns the operations associated with individual anchors and can compose them.
*   **Definition of complexity control:**
    *   Reducing the initialization scale or increasing weight decay limits the model's effective complexity.
    *   **Strong complexity control** forces the model to seek simpler, more general rules (reasoning) rather than memorize complex individual cases.
</details>


<details>
<summary><b>4. Validation on Real-World Tasks and Scaling Laws</b></summary>

*   **Task validation:** Experiments on Concept Graphs (compositional generalization), SCAN/COGS, and legal reasoning support the conclusion that small initialization improves generalization in reasoning.
*   **Insights from scaling laws under different initializations:**
    *   Controlling model complexity through small initialization can substantially lower the intercept of the test-error scaling curve.
    *   This means that, for the same amount of data or number of parameters, models with effective complexity control can achieve a higher performance ceiling.
</details>

## Chapter 14: Further Phenomena in Deep Neural Networks

### **Overview**

This chapter introduces a range of fascinating and counterintuitive deep learning phenomena beyond the Frequency Principle and condensation.

---

### **Outline**

<details>
<summary><b>1. Macroscopic Laws of Large Language Models</b></summary>

*   **Scaling law:**
    *   **Definition:** Model performance (loss) follows a power-law relationship with the number of model parameters ($N$), the amount of data ($D$), and the amount of computation ($C$): $L \propto X^{-\alpha}$.
    *   **Implication:** Increasing scale continues to improve performance, although with diminishing returns.
*   **Density law for large models:**
    *   **Definition:** Over time, newly released models require fewer parameters (effective parameter count) to achieve the same performance, meaning that model "density" grows exponentially.
    *   **Trend:** Model inference costs halve every 3-4 months, outpacing Moore's law.
</details>

<details>
<summary><b>2. Phenomena in Large Language Models</b></summary>

*   **In-context learning (ICL):**
    *   **Phenomenon:** Without updating their parameters, models can learn new tasks, such as sentiment analysis or image segmentation, from just a few examples in a prompt (few-shot learning).
    *   **Significance:** This capability resembles human learning and is a distinguishing feature of LLMs compared with traditional models.
*   **Chain of thought (CoT):**
    *   **Phenomenon:** Prompting models to generate intermediate reasoning steps, rather than directly output answers, substantially improves performance on complex tasks such as mathematics and logical reasoning.
    *   **Emergence:** CoT capabilities typically emerge only in sufficiently large models, such as those with more than 100 billion parameters.
</details>

<details>
<summary><b>3. Phenomena in Training Dynamics</b></summary>

*   **Grokking:**
    *   **Description:** On certain tasks, such as algorithmic computation, test error remains high (overfitting) after training error reaches 0. After prolonged further training, however, test error suddenly drops sharply and the model generalizes.
    *   **Implication:** Overfitting may be temporary; prolonged training can help a model move from memorization to understanding rules.
*   **Lottery ticket hypothesis:**
    *   **Hypothesis:** A randomly initialized dense network contains a sparse subnetwork (a winning ticket) that, when trained on its own, can match the performance of the original network.
    *   **Significance:** A network's large parameter count may simply make it easier to find this "winning" sparse structure.
*   **Double descent:**
    *   **Description:** As model complexity increases, test error first decreases, then increases (traditional overfitting), and then **decreases again**.
    *   **Conditions:** This phenomenon is especially pronounced in the presence of label noise, challenging the classical bias-variance trade-off.
</details>

<details>
<summary><b>4. Geometry of the Loss Landscape and Feature Space</b></summary>

*   **Neural collapse:**
    *   **Phenomenon:** Late in training, features of samples from the same class collapse to a single point (the class center) in feature space, while the centers of different classes form an equiangular configuration (ETF).
    *   **Significance:** This indicates that, near the end of training, the network learns to extract the most discriminative features while disregarding within-class differences.
*   **Mode connectivity:**
    *   **Phenomenon:** Different local minima (trained solutions) can be connected by a nonlinear path along which the loss remains low.
    *   **Implication:** Minima in the loss landscape are not isolated pits but lie on a connected low-loss manifold.
</details>

## Chapter 15: Solving Differential Equations with Neural Networks

### **Overview**

This chapter introduces **solving differential equations**, a core application of deep learning in scientific computing (AI for Science). Neural networks offer a new, mesh-free solution paradigm suited to high-dimensional problems, alongside traditional numerical methods such as finite differences and finite elements. An example based on Newton's laws of motion introduces the basic idea of reformulating differential equations as optimization problems. The chapter then details two major categories of methods: **parameterizing solutions** (such as PINN, Deep Ritz, and WAN) and **parameterizing operators** (such as DeepONet and FNO).

---

### **Outline**

<details>
<summary><b>1. The Basic Idea of Solving Differential Equations with Neural Networks</b></summary>

*   **Surrogate models and the optimization perspective:**
    *   Reformulate the differential-equation problem as an **optimization problem**, such as energy minimization.
    *   Use a neural network as a function approximator and train its parameters to minimize a residual or energy functional.
*   **Example: Newton's laws of motion**
    *   To solve for an object's position $s(t)$, construct a loss function containing the governing-equation residual $|m\frac{d^2s}{dt^2} - F|^2$ and the initial-condition error.
    *   Compute the loss and update network parameters using random, mesh-free sampling over the time domain.
</details>

<details>
<summary><b>2. Methods That Parameterize Solutions (Solving Instances)</b></summary>

*   **Least-squares method (PINN):**
    *   **Physics-informed neural networks:** Incorporate the PDE residual, boundary conditions, and initial conditions into the loss function as penalty terms.
    *   **Features:** Simple and intuitive; uses automatic differentiation to compute derivatives; requires no mesh; suitable for high-dimensional problems and complex domains.
*   **Variational method (Deep Ritz method):**
    *   **Principle:** Uses a variational principle to reformulate the PDE as the minimization of an energy functional, such as the Dirichlet energy.
    *   **Features:** Requires less solution smoothness (only first derivatives), offers better numerical stability, and is commonly used for elliptic equations.
*   **Weak-solution method (WAN):**
    *   **Principle:** Uses the weak (integral) formulation of a PDE and introduces an adversarial network as the test function.
    *   **Features:** Solves the problem through a min-max game and is suitable for solutions with low regularity, such as shocks.
</details>

<details>
<summary><b>3. Methods That Parameterize Operators (Learning Operators)</b></summary>

*   **Core objective:** Learn an operator mapping input functions, such as initial conditions or source terms, to output functions (solutions), rather than solve a single equation.
*   **Representative methods:**
    *   **DeepONet:** Learns the nonlinear mapping of an operator based on the universal approximation theorem.
    *   **Fourier neural operator (FNO):** Learns operators using integral transforms in the frequency domain and is resolution-independent.
*   **Advantage:** Once trained, inference is extremely fast, making these methods suitable for applications that repeatedly solve the same class of equations, such as inverse problems and real-time control.
</details>

<details>
<summary><b>4. Comparing Neural Networks with Traditional Methods: Frequency Bias</b></summary>

*   **Frequency Principle:**
    *   Neural networks tend to **learn low-frequency components first** and converge very slowly on high-frequency details, such as rapid oscillations or abrupt changes.
*   **Traditional numerical methods, such as Jacobi iteration:**
    *   They typically eliminate **high-frequency errors** quickly but are inefficient at eliminating low-frequency errors, which motivated the development of multigrid methods.
*   **Complementarity and integration:**
    *   **Hybrid strategy:** Use a neural network to quickly fit the low-frequency structure, then use a traditional method to correct high-frequency details.
    *   **Multiscale deep neural networks (MscaleDNN):** Introduce subnetworks at different scales to force the model to attend to high-frequency components, effectively addressing slow convergence at high frequencies.
</details>

<details>
<summary><b>5. Advantages and Limitations of Solving PDEs with Deep Learning</b></summary>

*   **Advantages:**
    *   **Mesh-free:** Avoids the difficulty of generating meshes for complex geometric domains.
    *   **High-dimensional capability:** Can effectively handle high-dimensional PDEs, such as the Schrödinger and Black-Scholes equations, overcoming the curse of dimensionality.
    *   **Inverse problems:** Conveniently unifies parameter inversion and forward solution within a single optimization framework.
*   **Limitations:**
    *   **Limited accuracy:** Struggles to achieve the high precision of traditional scientific computing, such as $10^{-10}$.
    *   **Lack of theoretical guarantees:** Comprehensive theoretical support for convergence and error bounds remains incomplete.
    *   **Training cost:** For a single equation, training takes much longer than traditional methods and may become trapped in local optima.
</details>

## Contact

For questions or further discussion, please contact [xuzhiqin@sjtu.edu.cn](mailto:xuzhiqin@sjtu.edu.cn).

