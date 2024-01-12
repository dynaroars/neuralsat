# **NeuralSAT**: A DPLL(T) Framework for Verifying Deep Neural Networks


*NeuralSAT* is a deep neural network (DNN) verification tool.  It integrates the DPLL(T) approach commonly used in SMT solving with a theory solver specialized for DNN reasoning. NeuralSAT exploits multicores and GPU for efficiency and can scale to networks with millions of parameters.  It also supports a wide range of neural networks and activation functions.


====================
<!-- 
**NeuralSAT** is a technique and prototype tool for verifying DNNs. 
It combines ideas from DPLL(T)/CDCL algorithms in SAT/SMT solving with a abstraction-based theory solver to reason about DNN properties. 
**NeuralSAT** takes as input the formula $\alpha$ representing the DNN `N` (with non-linear ReLU activation) and the formulae $\phi_{in}\Rightarrow \phi_{out}$ representing the property $\phi$ to be proved. 
Internally, **NeuralSAT** checks the satisfiability of the formula: $\alpha \land \phi_{in} \land \overline{\phi_{out}}$. **NeuralSAT** returns *`UNSAT`* if the formula is unsatisfiable, indicating  `N` satisfies $\phi$, and *`SAT`* if the formula is satisfiable, indicating the `N` does not satisfy $\phi$.

**NeuralSAT** uses a DPLL(T)-based algorithm to check unsatisfiability. 
It applies DPLL/CDCL to assign values to boolean variables and checks for conflicts the assignment has with the real-valued constraints of the DNN and the property of interest. 
If conflicts arise, **NeuralSAT** determines the assignment decisions causing the conflicts and learns clauses to avoid those decisions in the future. 
**NeuralSAT** repeats these decisions and checking steps until it finds a full assignment for all boolean variables, in which it returns *`SAT`*, or until it no longer can decide, in which it returns *`UNSAT`*. -->



## News
- NeuralSAT is ranked 4th in the recent VNN-COMP'23 (verify neural networks competition).  This was our first participation and we look forward to next time.

## Features

- **standard** input and output formats
  - input: `onnx` for neural networks and `vnnlib` for specifications
  - output: `unsat` for proved property, `sat` for disproved property (accompanied with a counterexample), and `unknown` for property that cannot be proved.
  

- **versatile**: support multiple types of neural types of networks and activation functions
  - layers (can be mixture of different types): `fully connected` (fc), `convolutional` (cnn), `residual networks` (resnet), `batch normalization` (bn)
  - activation functions:  `ReLU`, `sigmoid`, `tanh`, `power`

- **well-tested**
  - NeuralSAT has been tested on a wide-range of benchmarks (e.g., ACAS XU, MNIST, CFAR).
 
- **fast** and among the most scalable verification tools currently
  - NeuralSAT exploits and uses multhreads (i.e., multicore processing/CPUS) and GPUs available on your system to improve its performance.

- **active development** and **frequent updates**
  - If NeuralSAT does not support your problem, feel free to contact us (e.g., by [openning a new Github issue](https://arxiv.org/pdf/2307.10266.pdf)). We will do our best to help.
  - We will release new, stable versions about 3-4 times a year
  
- **fully automatic**, **ease of use** and requires very little configurations or expert knowledge
  - NeuralSAT requires *no* parameter tuning (a huge engineering effort that researchers often don't pay attention to)!  In fact, you can just apply NeuralSAT *as is* to check your networks and desired properties.  The user *does not* have to do any configuration or tweaking.  It just works!
    - But of course if you're an expert (or want to break the tool), you are welcome to tweak its internal settings.  
  - This is what makes NeuralSAT different from other DNN verifiers (e.g., AB-Crown), which require lots of tuning for the tools to work properly.

<details>

<summary><kbd>details</kbd></summary>

- **sound** and **complete** algorithm: will give both correct `unsat` and `sat` results
- combine ideas from conflict-clause learning (CDCL), abstractions (e.g., polytopes), LP solving
- employ multiple adversarial attack techniques for fast counterexamples (i.e., `sat`) discovery
</details>




## People
- Hai Duong (GMU, main developer)
- Linhan Li (GMU)
- ThanhVu Nguyen (GMU)
- Matt Dwyer (UVA)
- Dong Xu (UVA)


## :page_with_curl: Publications
- Hai Duong, Linhan Li, ThanhVu Nguyen, Matthew Dwyer, [**A DPLL(T) Framework for Verifying Deep Neural Networks**](https://arxiv.org/pdf/2307.10266.pdf), Arxiv, 2023

## Acknowledgements
The *NeuralSAT* research is partially supported by grants from NSF ([2238133](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2238133)) and an Amazon Research Award.

