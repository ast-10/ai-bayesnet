from variable import Variable
from typing import Dict, List
from probGraphicalModels import BeliefNetwork
from probStochSim import RejectionSampling
from probVE import VE
from probFactors import Prob
import json
from os import path
import timeit
import csv
import math


def perform_exact_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], ordering: List[Variable]) -> Dict[int, float]:
    """ Computes P(Q | E) on a Bayesian Network using variable elimination
        Arguments:
            model, the Bayesian Network
            Q, the query variable
            E, the evidence
            ordering, the order in which variables are eliminated
        
        Returns
            result, a dict mapping each possible value (q) of Q to the probability P(Q = q | E)
    """
    ve = VE(model)
    result = ve.query(var=Q, obs=E, elim_order=ordering)
    return result


def perform_approximate_inference(model: BeliefNetwork, Q:Variable, E: Dict[Variable, int], n_samples: int) -> Dict[int, float]:
    """
    Performs approximate inference using rejection sampling.
    
    Arguments:
        model: BeliefNetwork, the Bayesian Network.
        Q: Variable, the query variable.
        E: Dict[Variable, int], the evidence.
        n_samples: Number of samples to generate.
    
    Returns:
        A dictionary mapping values of Q to their approximate probabilities.
    """
    rs = RejectionSampling(model)
    result = rs.query(qvar=Q, obs=E, number_samples=n_samples)
    return result


def calculate_pac_sample_size(epsilon: float, delta: float) -> int:
    # should return 18445
    result = math.ceil(-math.log(delta / 2) / (2 * epsilon**2))
    print("\n PAC sample size: ", result)
    return result

def calculate_mse(exact: Dict[int, float], approx: Dict[int, float]) -> float:
    mse = 0
    for i in exact:
        mse += (exact[i] - approx[i]) ** 2
    result = mse / len(exact)
    return result


if __name__ == "__main__":
    
    variables = json.load(open(path.join("child", "variables.json")))
    tables = json.load(open(path.join("child", "tables.json")))
    name_to_node = {}
    nodes = []
    value_to_name = {}
    name_to_value = {}
    for variable in variables:
        variable_name = variable["name"]
        variable_values = variable["values"]
        variable_value_names = variable["value_names"]
        node = Variable(variable_name, variable_values)
        name_to_node[variable_name] = node
        nodes.append(node)

        value_to_name[variable_name] = {value: name for value, name in zip(variable_values, variable_value_names)}
        name_to_value[variable_name] = {name: value for value, name in zip(variable_values, variable_value_names)}

    cpts = []
    for table in tables:
        variable_name = table["variable"]
        node = name_to_node[variable_name]
        parent_names = table["parents"]
        parents = [name_to_node[parent_name] for parent_name in parent_names]
        probability_values = table["values"]
        cpt = Prob(node, parents, probability_values)
        cpts.append(cpt)

    bn = BeliefNetwork("child", nodes, cpts)

    Q = name_to_node["Disease"]
    E = {
        name_to_node["CO2Report"]: 1,
        name_to_node["XrayReport"]: 0,
        name_to_node["Age"]: 0
    }

    # perform Exact Inference for 2 different types of ordering
    # ordering 1 -- alphabetical ordering
    # ordering 2 -- LLM generated better ordering using Min-Fill

    alphabetical_order = [
    "Age", "BirthAsphyxia", "CO2", "CO2Report", "CardiacMixing", "ChestXray",
    "Disease", "DuctFlow", "Grunting", "GruntingReport", "HypDistrib",
    "HypoxiaInO2", "LVH", "LVHreport", "LowerBodyO2", "LungFlow",
    "LungParench", "RUQO2", "Sick", "XrayReport"
    ]

    better_order = [
    "GruntingReport", "RUQO2", "LowerBodyO2", "LVHreport", "BirthAsphyxia",
    "LungFlow", "Grunting", "ChestXray", "CO2", "DuctFlow", "HypDistrib",
    "LVH", "Sick", "CardiacMixing", "HypoxiaInO2", "LungParench", "Disease"
    ]

    alphabeticalOrder = [name_to_node[x] for x in alphabetical_order]
    betterOrder = [name_to_node[y] for y in better_order]


    # average time taken over 10 runs
    time_exact_alphabetical = timeit.timeit(lambda: perform_exact_inference(bn, Q, E, alphabeticalOrder), number=10) / 10
    time_exact_better = timeit.timeit(lambda: perform_exact_inference(bn, Q, E, betterOrder), number=10) / 10

    with open("part1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time_exact_alphabetical, time_exact_better])

    # perform Approximate Inference
    # given epsilon = 0.01, delta = 0.05
    # do for sample sizes -- 10, 100 and 18445
    sample_sizes = [10, 100, calculate_pac_sample_size(0.01, 0.05)]
    mse_alphabetical = []
    mse_better = []
    approx_times = []
    
    # calculating exact inference values to compare the results
    result_exact_alphabetical = perform_exact_inference(bn, Q, E, alphabeticalOrder)
    result_exact_better = perform_exact_inference(bn, Q, E, betterOrder)

    for n in sample_sizes:
        # timing for Approximate Inference
        time = timeit.timeit(lambda: perform_approximate_inference(bn, Q, E, n), number=10) / 10
        approx_times.append(time)

        # compute MSE -- doing for both ordering
        mse_alphabetical_val = 0
        mse_better_val = 0
        for i in range(10):
            approx = perform_approximate_inference(bn, Q, E, n)
            mse_alphabetical_val += calculate_mse(result_exact_alphabetical, approx)
            mse_better_val += calculate_mse(result_exact_better, approx)
        mse_alphabetical.append(mse_alphabetical_val / 10)
        mse_better.append(mse_better_val / 10)

    with open("part21.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(approx_times)

    with open("part22.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # saving the results only for alphabetical ordering since both generates the same results
        writer.writerow(mse_alphabetical)

    ## printing all outputs for better understanding ##

    print("\nResult of exact inference in alphabetical order:")
    for i in Q.domain:
        print(f"P({Q.name} = {value_to_name[Q.name][i]}) = {result_exact_alphabetical[i]:.6f}")

    print("\nResult of exact inference in better order:")
    for i in Q.domain:
        p = result_exact_better[i]
        print(f"P({Q.name} = {value_to_name[Q.name][i]}) = {p}")

    print("\nApproximate Inference Results:")
    for n in sample_sizes:
        approx_result = perform_approximate_inference(bn, Q, E, n)
        print(f"\nSample size {n}:")
        for i in Q.domain:
            print(f"P({Q.name} = {value_to_name[Q.name][i]}) = {approx_result[i]:.6f}")


    print("\nExact Inference Times: Alphabetical = {:.6f}, Dynamic = {:.6f}".format(time_exact_alphabetical, time_exact_better))
    print("Approximate Inference Times:", approx_times)
    print("Approximate Inference Errors (MSE for Alphabetical Ordering): ", mse_alphabetical)
    print("Approximate Inference Errors (MSE for Better Ordering): ", mse_better)
