"""Sampling setup and model create/train helpers.

Training helpers adapted from HydroDL (https://github.com/mhpi/hydroDL).
Copyright (c) 2020, The Pennsylvania State University ("PSU").
See THIRD_PARTY_LICENSES/hydroDL-LICENSE.txt (Non-Commercial).
"""

from pathlib import Path
import torch.nn

from model import CudnnLstmModel, CudnnLstmModel2
from sampling import (
    build_numpy_training_tensors,
    build_numpy_test_tensors,
    no_iter_nt_ngrid,
    save_train_sampling_meta,
    take_sample_train,
)


class create_NN_models(torch.nn.Module):
    def __init__(self, args):
        super(create_NN_models, self).__init__()
        self.args = args
        self.get_model()

    def get_model(self) -> None:
        self.nx = len(self.args["forcings"] + self.args["attributes"])


        self.ny = len(self.args["targets"])


        if self.args["NN_model_name"] == "LSTM":
            self.NN_model = CudnnLstmModel(nx=self.nx,
                                           ny=self.ny,
                                           hiddenSize=self.args["hidden_size"],
                                           args=self.args,
                                           dr=self.args["dropout"],
                                           )
        elif self.args["NN_model_name"] == "LSTM2":
            self.NN_model = CudnnLstmModel2(nx=self.nx,
                                            ny=self.ny,
                                            hiddenSize=self.args["hidden_size"],
                                            args=self.args,
                                            dr=self.args["dropout"],
                                            )
        else:
            print("NN model type has not been defined")
            exit()

    def breakdown_params(self, params_all):
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny_hydro]
        params_temp_model = params_all[:, :, self.ny_hydro: (self.ny_hydro + self.ny_temp)]


        if self.args['hydro_model_name'] != "None":

            params_dict["hydro_params_raw"] = torch.sigmoid(
                params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.args["nmul"]]).view(
                params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
                self.args["nmul"])

            if self.args["routing_hydro_model"] == True:
                params_dict["conv_params_hydro"] = torch.sigmoid(
                    params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.args["nmul"]:])
            else:
                params_dict["conv_params_hydro"] = None

        if self.args['temp_model_name'] != "None":

            if self.args["lat_temp_adj"] == True:
                params_dict["temp_params_raw"] = torch.sigmoid(
                    params_temp_model[:, :,
                    :(len(self.temp_model.parameters_bound) + len(self.temp_model.lat_adj_params_bound)) * self.args[
                        "nmul"]]).view(
                    params_temp_model.shape[0], params_temp_model.shape[1],
                    len(self.temp_model.parameters_bound) + len(self.temp_model.lat_adj_params_bound),
                    self.args["nmul"])

            else:
                params_dict["temp_params_raw"] = torch.sigmoid(
                    params_temp_model[:, :, :len(self.temp_model.parameters_bound) * self.args["nmul"]]).view(
                    params_temp_model.shape[0], params_temp_model.shape[1], len(self.temp_model.parameters_bound),
                    self.args["nmul"])

            if self.args["routing_temp_model"] == True:
                params_dict["conv_params_temp"] = torch.sigmoid(params_temp_model[-1, :, -len(self.temp_model.conv_temp_model_bound):])
            else:
                print("it has not been defined yet what approach should be taken in place of conv")
                exit()
        return params_dict


    def forward(self, dataset_dictionary_sample):
        y_sim = self.NN_model(dataset_dictionary_sample["inputs_NN_scaled"])
        return y_sim

def prepare_train_sampling(scaled_data: dict, args: dict, base_dir: Path) -> dict:


    dataset_dict, dnp_new = build_numpy_training_tensors(scaled_data, args, base_dir)
    scaled_data["train_dataset"] = dataset_dict
    scaled_data["D_N_P_processed"] = dnp_new

    x = dataset_dict["inputs_NN_scaled"]
    ngrid, n_iter_ep, nt_new, batch_size, dnp_for_iter = no_iter_nt_ngrid(
        "t_train", args, x, dnp_new
    )
    meta = {
        "ngrid": int(ngrid),
        "nIterEp": int(n_iter_ep),
        "nt_tensor": int(x.shape[0]),
        "nt_new_avg_days": float(nt_new),
        "batch_size": int(batch_size),
        "nx_features": int(x.shape[2]),
    }
    scaled_data["train_sampling_meta"] = meta
    save_train_sampling_meta(args["out_dir"], meta)
    return scaled_data


def prepare_test_sampling(scaled_data: dict, args: dict, base_dir: Path) -> dict:
    """Build ``test_dataset`` from gap-truth ``test_forcing`` over ``t_test``."""
    scaled_data["test_dataset"] = build_numpy_test_tensors(scaled_data, args, base_dir)
    return scaled_data
