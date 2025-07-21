
from physics.simulation import mcfm
from datasets.balanced import BalancedDataModule
from models.carl import CARL
import lightning as L

# input numerator & denominator hypothesis datasets
numerator_file = 'numerator.csv'
denominator_file = 'denominator.csv'

# make sure the features are loaded in as observables (i.e. kinematics)
features = ["l1_pt", "l1_eta", "l1_phi", "l1_energy", "l2_pt", "l2_eta", "l2_phi", "l2_energy", "l3_pt", "l3_eta", "l3_phi", "l3_energy", "l4_pt", "l4_eta", "l4_phi", "l4_energy"]
events_n = mcfm.from_csv(file_path = numerator_file, kinematics = features)
events_d = mcfm.from_csv(file_path = denominator_file, kinematics = features)

# for each element of ensemble, want to train
ensemble_size = 10
seeds = range(ensemble_size)
for seed in seeds:

    # Q: how to resample events for wifi ensemble?
    events_n_i = events_n.resample(random_state = seed)  # IMPLEMENT Process.resample()
    events_d_i = events_d.resample(random_state = seed)  # IMPLEMENT

    # IMPORTANT: force the random_state of dataset splitting to be the same so ensemble members always see the same training/validation data 
    # Q: should NN weights & biases be initialized the same way across an ensemble?
    ds_balanced = BalancedDataModule(numerator_events = events_n_i, denominator_events = events_d_i, features = features, batch_size=1024, random_state = 42)
    model_carl = CARL(n_features = len(features), n_layers = 20, n_nodes = 1024, learning_rate = 1e-3)

    trainer = L.Trainer(accelerator='gpu', devices=1)
    trainer.fit(
        model = model_carl,
        datamodule = ds_balanced
    )
