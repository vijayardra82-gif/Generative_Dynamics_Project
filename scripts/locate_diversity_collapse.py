import sys
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.timegan import Generator, Supervisor, Recovery

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

CHECKPOINTS = {
    "Experiment 8": OUTPUT_DIR / "lorenz_timegan_experiment8.pt",
    "Experiment 9": OUTPUT_DIR / "lorenz_timegan_experiment9.pt",
}

NUM_SAMPLES = 500
SEQ_LENGTH = 100

DEVICE = torch.device("cpu")


def calculate_diversity(data):
    return data.std(dim=0, unbiased=False).numpy()


for name, checkpoint_path in CHECKPOINTS.items():

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    hidden_dim = checkpoint["hidden_dim"]
    noise_dim = checkpoint["noise_dim"]
    input_dim = checkpoint["input_dim"]

    generator = Generator(noise_dim, hidden_dim).to(DEVICE)
    supervisor = Supervisor(hidden_dim).to(DEVICE)
    recovery = Recovery(hidden_dim, input_dim).to(DEVICE)

    generator.load_state_dict(checkpoint["generator"])
    supervisor.load_state_dict(checkpoint["supervisor"])
    recovery.load_state_dict(checkpoint["recovery"])

    generator.eval()
    supervisor.eval()
    recovery.eval()

    torch.manual_seed(42)

    noise = torch.randn(
        NUM_SAMPLES,
        SEQ_LENGTH,
        noise_dim,
        device=DEVICE
    )

    with torch.no_grad():

        generated_hidden = generator(noise)

        supervised_hidden = supervisor(generated_hidden)

        generated_output = recovery(supervised_hidden)

    stages = {
        "Input Noise": noise,
        "Generator Output": generated_hidden,
        "Supervisor Output": supervised_hidden,
        "Recovery Output": generated_output,
    }

    for stage_name, values in stages.items():

        diversity = calculate_diversity(values)

        print(f"\n{stage_name}")

        print("Initial diversity:", diversity[0])
        print("Middle diversity:", diversity[49])
        print("Final diversity:", diversity[-1])

        print(
            "Average diversity:",
            np.mean(diversity)
        )

print("\nDiversity localization completed!")