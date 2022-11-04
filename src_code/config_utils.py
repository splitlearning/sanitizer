import os
import json


def load_config_as_dict():
    """
    Load the contents of the config file as a dict object
    """
    base_path = os.path.dirname(__file__)
    rel_path = "config_utkface.json"
    rel_path = "config.json"
    path = os.path.join(base_path, rel_path)
    json_dict = None

    with open(path) as json_file:
        json_dict = json.load(json_file)

    return json_dict


def load_config(offline=False):
    """
    Properly load and configure the json object into the expected config format. This includes
    the calculations of dynamic variables.
    """
    json_dict = load_config_as_dict()
    experiment_dict = json_dict.get("experiment_config", {}).copy()
    research_dict = json_dict.get("research_config", {}).copy()
    research_dict.update(experiment_dict)
    json_dict = research_dict

    method = json_dict.get("method")
    dataset = json_dict.get("dataset")
    logits = json_dict.get("logits")
    attribute = json_dict.get("attribute")
    #attribute_serialized = "_".join(attribute)
    seed = json_dict.get("seed")

    if method == "ours":
        beta = json_dict.get("beta")
        nz = json_dict.get("nz")
        priv_dim = json_dict.get("priv_dim")
        dcorr = json_dict.get("dcorr")
        adv = json_dict.get("adv")
        mechanism = json_dict.get("mechanism") or "suppression"
        json_dict["mechanism"] = mechanism
    elif method in ["tiprdc", "adversarial", "maxentropy", "gap"]:
        _lambda = json_dict.get("lambda")
    elif method == "noise":
        sigma = json_dict.get("sigma")
    elif method == "vae":
        beta = json_dict.get("beta")

    experiment_name = "{}_".format(method)
    if method == "ours":
        if dcorr:
            experiment_name += "dcorr{}_".format(json_dict["alpha_values"][3])
        if adv:
            experiment_name += "adv{}_".format(json_dict["alpha_values"][2])
        experiment_name += "{}_beta{}_{}_new_z{}".format(dataset, beta, attribute, nz)
    elif method in ["tiprdc", "adversarial", "maxentropy", "gap"]:
        experiment_name += "{}_lambda{}_{}".format(dataset, _lambda, attribute)
        if method == "gap":
            experiment_name += "_D{}".format(json_dict.get("D"))
    elif method == "noprivacy":
        experiment_name += "{}_{}".format(dataset, attribute)
    elif method == "noise":
        experiment_name += "{}_{}_{}".format(dataset, sigma, attribute)
    elif method == "vae":
        experiment_name += "beta{}_{}".format(beta, attribute)
    else:
        print("method {} not implemented".format(method))
        exit()


    if offline:
        if method == "ours" and mechanism:
            experiment_name += "{}".format(mechanism)
            if mechanism == "obfuscation" or mechanism == "dpgmm":
                experiment_name += "_eps{}".format(json_dict.get("eps"))
        experiment_name += "_offline"
        if json_dict.get("is_cas"):
            experiment_name += "_cas"

    experiment_name += "_seed{}".format(seed)

    json_dict["experiment_name"] = experiment_name

    return json_dict
