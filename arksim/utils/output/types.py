from enum import Enum


class OutputDir(str, Enum):
    SCENARIO = "scenario"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    RESULTS = "results"
