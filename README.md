# Measures for Estimating Unfairness in Data under Differential Privacy

Authors: Amir Gilad, Mariia Vologdin, Yuchao Tao


This repository contains code accompanying our work on the paper **Measuring Unfairness Through Dependency Quantification Under Differential Privacy**. The repository includes implementations of several unfairness measures and a comprehensive experimental pipeline evaluating their **scalability**, **accuracy under DP**, and **behavior under increasing unfairness**, using both real-world and synthetic datasets.

---

## Abstract of the Paper
Differential privacy (DP) has become the de facto standard for protecting sensitive data, providing strong guarantees 
that published statistics or models reveal limited information about any individual. However, privacy noise and 
restricted data access make it increasingly difficult to assess the fairness and reliability of private datasets.
In this paper, we propose a formal framework for quantifying data unfairness under DP. We identify three core desiderata 
for unfairness measures based on previous work: positivity, monotonicity, and DP computability. We further instantiate 
them through three complementary measures: (1) a mutual information–based measure with a total variation distance proxy 
suitable for DP, (2) a data-repair–based measure approximated via a reduction to weighted MaxSAT, and (3) a top-𝑘 tuple 
contribution measure that isolates the most influential records in fairness violations. We design privacy-preserving 
algorithms and analyze their sensitivity, accuracy, and efficiency. Extensive experiments on multiple real-world 
datasets demonstrate that our proposed measures faithfully approximate their non-private counterparts, effectively 
quantify bias under privacy constraints, and provide insights for fair data management.

## Full Version of the Paper
As well as the code, this repository contains the full version of our paper, titled `full_version.pdf`.

## Implemented Measures
The following unfairness measures and proxies are implemented:

- **Proxy Mutual Information (TVD)**  
  `ProxyMutualInformationTVD`
- **Proxy Repair via MaxSAT**  
  `ProxyRepairMaxSat`
- **Tuple Contribution**  
  `TupleContribution`
- **Baseline (non-proxy)** Mutual Information  
  `MutualInformation`
- **(Unused)** PrivBayes proxy - also used as a baseline, taken from another paper
  `ProxyMutualInformationPrivbayes`
- **(Unused)** Proxy Mutual Information Lipschitz 
  `ProxyMutualInformationLipschitz`
- **(Unused)** Proxy Mutual Information NIST Contest
  `ProxyMutualInformationNistContest`
- **(Unused)** PMI Threshold Detector
   `PMIThresholdDetector`
- **(Unused)** Anomalous Treatment Count
   `AnomalousTreatmentCount`
- **(Unused)** Layered Shapley Values
   `LayeredShapleyValues`

You can find the unused measures under `unused_measures\`.

All the used measures expose a common interface:
```
m = MeasureClass(data=df)
value = m.calculate(fairness_criteria, epsilon=epsilon)
```

## Repository structure
```
.
├── data/                      # Input datasets (CSV files)
├── plots/                     # Generated plots
├── unused_measures/           # Implementations of unused measures
├── requirements.txt           # Required packages list
├── mutual_information.py
├── proxy_mutual_information_tvd.py
├── proxy_repair_maxsat.py
├── tuple_contribution.py
├── main.py                    # Experiment runners and plotting code
└── full_version.pdf           # Full version of the paper
```

## Experiments
All the experiments to generate all the plots in the paper are placed in the main.py file. To run them, simply install 
all the dependencies from `requirements.txt` and run `main.py`.

## License
The code is licensed under Apache 2.0 .
