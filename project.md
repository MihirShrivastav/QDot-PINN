Project Documentation: Eigenstate Solver for Quantum Dots using PINNs

Abstract

This project aims to develop a computational tool to solve the 2D time-independent Schrödinger equation for an electron confined within a quantum dot of arbitrary potential. We will leverage a Physics-Informed Neural Network (PINN) to find the low-lying energy eigenvalues (En​) and their corresponding wavefunctions (ψn​). The PINN framework is uniquely suited for this problem as it doesn't require a mesh and can handle complex potential geometries. The final output will be a validated model that can accurately predict the quantum states, a cornerstone for designing and understanding nano-electronic devices.

1. The Problem to Solve

In nanotechnology, a quantum dot is a tiny semiconductor crystal that confines an electron in all three dimensions. From a quantum mechanics perspective, this is a "particle in a box" problem, but for realistic devices, the "box" has a complex shape and soft, finite potential walls.

The core physical problem is to solve the time-independent Schrödinger equation, which is an eigenvalue problem:
H^ψ(x,y)=Eψ(x,y)


Where:

    H^=−2mℏ2​∇2+V(x,y) is the Hamiltonian operator.

    ψ(x,y) is the wavefunction, whose squared magnitude ∣ψ∣2 represents the probability of finding the electron at position (x,y).

    E is the energy eigenvalue, representing the quantized energy level of the state.

    V(x,y) is the confining potential of the quantum dot.

The computational challenge is that for an arbitrary potential V(x,y), this equation has no analytical solution. Traditional numerical methods like the Finite Element Method (FEM) require complex mesh generation. Our goal is to bypass this by using a mesh-free PINN approach.

2. The Approach: PINN as an Eigenvalue Solver

Instead of using a traditional numerical solver, we will model the wavefunction using a neural network.

    Represent the Wavefunction: A neural network ΨNN​(x,y;θ) with weights and biases θ will be used to approximate the true wavefunction ψ(x,y). The inputs are the coordinates (x,y), and the output is the value of the wavefunction.

    Treat Energy as a Trainable Parameter: The energy eigenvalue E will be initialized as a trainable scalar variable, which is optimized directly during training, just like the network's weights.

    Enforce Physics via the Loss Function: The network is not trained on data in the traditional sense. Instead, it's trained to satisfy the physical laws at a large number of randomly sampled points ("collocation points"). The loss function will be composed of three parts:

        PDE Residual Loss (LPDE​): This is the main part. It punishes the network if its output doesn't satisfy the Schrödinger equation. We define the residual as:
        R(x,y)=(−2mℏ2​∇2+V(x,y))ΨNN​−E⋅ΨNN​


        The loss is the mean squared error of this residual over all collocation points.

        Boundary Condition Loss (LBC​): For a confined electron, the wavefunction must vanish far away from the dot. We enforce ΨNN​(x,y;θ)=0 at the boundaries of our simulation domain.

        Normalization Loss (Lnorm​): The total probability of finding the electron must be 1. This means the wavefunction must be normalized: ∫Ω​∣ΨNN​∣2dxdy=1. This prevents the trivial solution ψ=0.

The total loss is a weighted sum: L=λPDE​LPDE​+λBC​LBC​+λnorm​Lnorm​.

3. Model and Architecture

    Neural Network: A standard Multi-Layer Perceptron (MLP) is sufficient.

        Input Layer: 2 neurons (for x and y coordinates).

        Hidden Layers: 5-8 hidden layers with 64-128 neurons each. This depth is needed to learn the complex, oscillatory patterns of wavefunctions.

        Activation Function: Hyperbolic Tangent (tanh) or swish. These are smooth, continuously differentiable functions, which is essential because we need to compute the second derivative (Laplacian ∇2) for the PDE residual.

        Output Layer: 1 neuron (for the value of ψ).

    Potential V(x,y): We can start with a simple case and move to a more interesting one.

        Asymmetric "Soft" Box: A potential that is zero in a rectangular region and rises smoothly (not infinitely) at the edges. This is more realistic than the textbook "infinite square well."
        V(x,y)=V0​(1+e−k(x−Lx​)1​+1+ek(x+Lx​)1​+…)

        Double Quantum Dot: Two potential wells close to each other. This is exciting because it allows you to find "bonding" and "anti-bonding" states, demonstrating quantum tunneling.

4. Training Data

This is a key difference with standard machine learning. We do not need a pre-existing dataset of solved wavefunctions.

The "training data" consists of collocation points that we generate ourselves:

    Domain Points (NPDE​): ~10,000 points sampled randomly from within the simulation domain (e.g., a square [−2,2]×[−2,2]). Latin Hypercube Sampling is the preferred method for this as it ensures a more uniform coverage of the space. These points are used to compute LPDE​.

    Boundary Points (NBC​): ~2,000 points sampled randomly on the boundary of the domain. These are used to compute LBC​.

    Normalization Points (Nnorm​): The normalization integral will be computed numerically using a large grid of points (~100x100) covering the domain.

5. Expected Results and Validation

The primary outputs of the project will be:

    Energy Eigenvalues: A list of the calculated energy levels for the ground state (E0​) and the first few excited states (E1​,E2​,…).

    Wavefunction Plots: For each energy level, a 2D heatmap or contour plot of the corresponding probability density ∣ψn​(x,y)∣2. We expect to see the classic shapes: a single lobe for the ground state, two lobes for the first excited state, etc.

Validation: How do we know the results are correct?

    Compare with a Traditional Solver: We will solve the same problem using an established numerical method (e.g., a Finite Difference Method script or a FEniCS/COMSOL model). We will then compare the energy eigenvalues and the overall shape of the wavefunctions. The PINN results should match the validated solver's results within a small tolerance.