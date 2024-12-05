import os
import sys
import importlib.util
from functools import partial
import click
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
import deepqmc
from deepqmc.train import train
from deepqmc.sampling import combine_samplers, DecorrSampler, MetropolisSampler, initialize_sampling
from deepqmc.hamil import MolecularHamiltonian,Molecule
from deepqmc.log import H5Logger


def run(ansatz_func, save_path, steps, electron_batch_size, seed):

    h5_logger_constructor=partial(H5Logger, keys_whitelist=['local_energy', 'time','r'])

    mol = Molecule(  # LiH
        coords=[[0.0, 0.0, 0.0], [3.015, 0.0, 0.0]],
        charges=[3, 1],
        charge=0,
        spin=0,
        unit='bohr',
    )

    H = MolecularHamiltonian(mol=mol)
    ansatz = ansatz_func(H)

    elec_sampler = partial(combine_samplers, samplers=[DecorrSampler(length=20), partial(MetropolisSampler)])
    sampler_factory = partial(initialize_sampling, elec_sampler=elec_sampler)

    deepqmc_dir = os.path.dirname(deepqmc.__file__)
    config_dir = os.path.join(deepqmc_dir, 'conf/task/opt')

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        # cfg = compose(config_name='kfac')
        # cfg = compose(config_name='kfac')
        cfg = compose(config_name='adamw')

    kfac = instantiate(cfg, _recursive_=True, _convert_='all')

    train(
        H, ansatz, kfac, sampler_factory,
        steps=steps, electron_batch_size=electron_batch_size, seed=seed,
        workdir=save_path,
        h5_logger_constructor=h5_logger_constructor
    )


@click.command()
@click.argument('python_file')
@click.option('--output_directory', default=None, help='Directory where the output will be saved.')
@click.option('--steps', default=500, type=int, help='Number of training steps. Default is 500.')
@click.option('--electron-batch-size', default=2000, type=int, help='Batch size for electrons. Default is 2000.')
@click.option('--seed', default=42, type=int, help='Random seed for reproducibility. Default is 42.')
def main(python_file, output_directory, steps, electron_batch_size, seed):
    """
    Run the quantum chemistry workflow.

    \b
    PYTHON_FILE: Path to the Python file defining the `initialize_wf` instance.
    OUTPUT_DIRECTORY: Directory where the output will be saved.
    """

    # Set default output directory if not provided
    if output_directory is None:
        output_directory = f"{python_file}.o"

    # Process the input Python file
    if python_file.startswith("builtin"):
        params = python_file.split(".")[1]
        python_file = "built_in.py"
    else:
        python_file = python_file + '.py'
        params = None

    # Validate the Python file path
    if not os.path.isfile(python_file):
        click.echo(f"Error: The specified file '{python_file}' does not exist.", err=True)
        sys.exit(1)

    # Validate or create the output directory
    if not os.path.isdir(output_directory):
        try:
            os.makedirs(output_directory)
            click.echo(f"Output directory '{output_directory}' created.")
        except Exception as e:
            click.echo(f"Error: Failed to create output directory '{output_directory}'. {e}", err=True)
            sys.exit(1)

    # Dynamically load the Python file
    module_name = os.path.splitext(os.path.basename(python_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, python_file)
    if spec is None:
        click.echo(f"Error: Unable to create a module spec for '{python_file}'.", err=True)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        click.echo(f"Error: Failed to execute the module '{python_file}'. {e}", err=True)
        sys.exit(1)

    # Access the `initialize_wf` instance
    if not hasattr(module, "initialize_wf"):
        click.echo(f"Error: The file '{python_file}' does not contain an 'initialize_wf' instance.", err=True)
        sys.exit(1)

    initialize_wf = getattr(module, "initialize_wf")

    if params is not None:
        initialize_wf = partial(initialize_wf, config_name=params)

    # Initialize the wave function
    run(initialize_wf, output_directory, steps, electron_batch_size, seed)


if __name__ == "__main__":
    main()
