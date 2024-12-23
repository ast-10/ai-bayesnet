from probVE import VE
from probStochSim import RejectionSampling 
from probFactors import Variable, Prob
from probGraphicalModels import BeliefNetwork
from os import path 
import json


variables = json.load(open(path.join("earthquake", "variables.json")))

name_to_node = {}
nodes = []
value_to_name = {}
name_to_value = {}
for variable in variables:
    variable_name = variable["name"]
    variable_values = variable["values"]
    variable_value_names = variable["value_names"]
    node =  Variable(variable_name, variable_values)
    name_to_node[variable_name] = node 
    nodes.append(node)

    value_to_name[variable_name] = { value: name for value, name in zip(variable_values, variable_value_names)}
    name_to_value[variable_name] = { name: value for value, name in zip(variable_values, variable_value_names)}

tables = json.load(open(path.join("earthquake", "tables.json")))
cpts = []
for table in tables:
    variable_name = table["variable"]
    node = name_to_node[variable_name]
    parent_names = table["parents"]
    parents = [name_to_node[parent_name] for parent_name in parent_names]
    probability_values = table["values"]
    cpt = Prob(node, parents, probability_values)
    cpts.append(cpt)

bn = BeliefNetwork("earthquake", nodes, cpts)
exact_infer = VE(gm=bn)
VE.max_display_level = -1


Q =  "Alarm"
E = {}
# VE Computes P(Q | E)
result = exact_infer.query(var=name_to_node[Q], obs=E, elim_order=nodes)
for i in name_to_node[Q].domain:
    p = result[i]
    print (f"P({Q} = {value_to_name[Q][i]}) = {p}")

approx_infer = RejectionSampling(bn)
result = approx_infer.query(qvar=name_to_node[Q], obs=E, number_samples=100)

for i in name_to_node[Q].domain:
    p = result[i]
    print (f"P({Q} = {value_to_name[Q][i]}) = {p}")

