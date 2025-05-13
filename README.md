# SelfOrganizingMaps
#### Self Organizing Maps (Kohonen Networks) implemented from scratch. 
#### This project focuses on implementing a **Self Organizing Maps (SOM)** from scratch using **NumPy**.
---

## 🎯 Project Overview

This laboratory project dives deep into the methods of unsupervised learning, with an emphasis on the SOM model, one of the popular methods for 2d representation of multidimensional data. The primary goal was not only to understand how SOMs work internally but also to:

- 🛠️ Build a functional SOM from scratch.
- 🔬 Explore and analyze the influence of hyperparameters on learning.
- 📊 Validate theoretical knowledge through hands-on experiments and visualizations.

As part of the lab course, we proceed through a development*of a SOM — from grids (Square or Hexagonal) to neighbourhood functions (Gaussian or Mexican Hat), training, evaluation, and visualization — all implemented using **only NumPy**.

---

## 🧪 Laboratory Progress & Notebook Summary

Each Jupyter Notebook (`KOH1.ipynb`, `KOH2.ipynb` and `report_notes.ipynb`) reflects a major milestone in the development process.

| Notebook | Topic | Highlights |
|----------|-------|------------|
| `KOH1.ipynb` | 🟥 Square Grid  | Initial experiments with **square grid** and simple datasets (Hexagon and Cube). |
| `KOH2.ipynb` | 🛑 Hexagonal Grid | Experiments with **hexagonal grid** and simple datasets (MNIST and Human Activity Recognition Using Smarophones). |
| `report_notes.ipynb` | ⚡ Further Experiments and Evaluations  | Compared **RMSProp** and **Momentum** optimizers |


---

## 📁 Repository Structure & Description

- **`data/`**  
  Contains datasets used for training and evaluation.

- **`model/`**  
  Core components for building and training the MLP model.
  - `network.py`: Contains SOM implementation with visualisation methods.
  - `distance.py`: Defines distance functions..
    
- **`results/`**  
  Placeholder for experiments results.

- **`KOHx.ipynb`**  
  Examples and experiments.

- **`report_notes.ipynb`**  
  Draft notebook for report with extensive experiments, evaluations and visualizations.

- **`report.pdf`**
  Final report including experiments, results analysis and comparison.

- **`README.md`**  
  This documentation file.


---

## 📌 Key Takeaways

- ✅ **Implemented from scratch**: All components such as distance functions, network model and training loops were developed manually.
- 🔍 **Empirical Analysis**: Rich set of experiments to explore how various parameters affect learning.
- 📈 **Visualization & Reporting**: Comprehensive visual tools to interpret Kohonen Network performance.

---

## 🧾 Final Notes

This project is a **learning-focused implementation** — designed to understand how SOMs work under the hood. Feel free to explore and experiment! 🚀

---

> Created as part of an academic laboratory course on Computational Intelligence Methods in Data Analysis (Metody Inteligencji Obliczeniowej w Analizie Danych), a part of Data Science Bachelor Engineering programme at Warsaw University of Technology.
