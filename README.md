# Parameter-Extraction-for-Superconducting-Filter
The purpose of this program is to extract the M-coupling matrix from simulated S-parameters. The input training dataset consists of paired S-parameters and corresponding M-matrices, where the S-parameters are generated from the M-matrices using analytical filter synthesis formulas. This approach is known as inverse design. Once the model is trained, it can be used to extract the M-matrix corresponding to S-parameters obtained from full-wave simulations. This enables rapid identification of discrepancies between the extracted and target M-matrices, thereby guiding the direction of manual parameter tuning.

Compared to the more accurate but computationally intensive Cauchy method, the proposed extraction algorithm has a simpler structure and uses only the magnitude of S₁₁ as input. This significantly reduces the model’s input dimensionality and, consequently, the training time.

Key innovations of the model include:

Fourier transformation of input S-parameters: The input S-parameter data is transformed into the frequency domain via the fast Fourier transform (FFT). This enriches the feature space by highlighting high-frequency components and bandwidth-related characteristics, enabling more precise representation of resonant behavior and spectral shape.

Incorporation of a self-attention mechanism: This allows the model to dynamically weigh different parts of the input sequence, capturing long-range dependencies and intrinsic correlations within a single data sample. As a result, the network can focus on critical spectral features—such as resonance dips and transition bands—improving both accuracy and interpretability.
For detailed training results and performance evaluation, please refer to the manuscript currently under review: Linear Phase Superconducting Bandpass Filter Optimization Accelerated with Enhanced Multilayer Perceptron for Parameter Extraction, submitted to Supercondutor Science and Technology
