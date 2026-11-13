# Generative Dynamics Project



## Project Overview



This project investigates the generation of synthetic time-series data using TimeGAN (Time-series Generative Adversarial Network).



The main objective is to generate synthetic trajectories that capture important statistical and temporal characteristics of chaotic dynamical systems, with particular focus on the Lorenz system.



The project also includes experiments involving the Rossler and Duffing systems.



## Technologies Used



- Python

- PyTorch

- NumPy

- Pandas

- SciPy

- Matplotlib

- TimeGAN

- GRU (Gated Recurrent Unit)



## Project Structure



- `data/` â€“ Original datasets and prepared time-series windows.

- `models/` â€“ TimeGAN model implementation.

- `scripts/` â€“ Dataset generation, model training, synthetic generation, evaluation and diagnostic scripts.

- `outputs/` â€“ Model checkpoints, synthetic datasets, evaluation results and comparison plots.

- `notebooks/` â€“ Jupyter notebooks.

- `requirements.txt` â€“ Python dependencies.

- `README.md` â€“ Project documentation.



## Methodology



1\. Generate and prepare Lorenz, Rossler and Duffing datasets.

2\. Divide time series into fixed-length windows.

3\. Implement the TimeGAN architecture using GRU-based neural networks.

4\. Train the model through reconstruction, supervisor and adversarial training phases.

5\. Generate synthetic time-series trajectories.

6\. Evaluate synthetic data against real data.

7\. Compare results across multiple experiments.



## Model Architecture



The TimeGAN implementation contains five main components:



- Embedder

- Recovery Network

- Generator

- Supervisor

- Discriminator



The model uses recurrent GRU layers to learn temporal patterns in sequential data.



## Experiments



Multiple training experiments were conducted with different configurations and loss-weight settings.



Experiments 4 to 9 were trained and evaluated. Experiment 10 contains an additional training configuration and has not yet been trained.



## Evaluation Methods



The project includes:



- Statistical distribution comparison

- Wasserstein distance

- Autocorrelation analysis

- Cross-correlation analysis

- Spectral analysis

- Phase-space comparison

- Trajectory comparison

- Physics residual evaluation

- Initial-state diversity analysis

- Lobe transition analysis



## Installation



Clone the repository:



```bash

git clone https://github.com/vijayardra82-gif/Generative\_Dynamics\_Project.git

cd Generative\_Dynamics\_Project

```



Install dependencies:



```bash

pip install -r requirements.txt

```



## Running the Project



Run scripts from the project root directory.



For example:



```bash

python scripts/generate\_lorenz.py

```



Training, generation and evaluation scripts are available in the `scripts/` directory.



## Results



The `outputs/` directory contains:



- Trained model checkpoints

- Synthetic Lorenz trajectories

- Evaluation result files

- Phase-space comparison plots

- Trajectory comparison plots

- Temporal and spectral analysis results



The experiments show improvements in certain evaluation metrics. However, the generated trajectories do not fully reproduce all the dynamical properties of the real Lorenz system.



## Limitations



Initial-state diversity, lobe-transition behaviour and physics residuals remain areas for improvement.



The generated trajectories do not yet reproduce the real Lorenz attractor geometry convincingly.



## Project Status



Experiments 4â€“9 have been trained and evaluated. Experiment 10 is an additional, untrained experimental configuration.


