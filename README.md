# RL_2024_PA_3

## The code strcuture

* `Assignment.ipynb` contains all the implementations for the options and models necessary for the assignment

    * Here, options were learned to move a car to a particular location

---

* A variation of options were later learned and have showm to perform better
    * These options also encapsulate picking up or dropping off the passenger
    * We also encode the state space to only contain passenge and driver locations, to enable faster learning 
    * The files have been split for this work, in such a manner:
        * The algorithms can be found in `algorithms.py`
        * The option can be found in `options.py`
        * The analysis can be found in `taxi_v3_PA_options_variant.ipynb`


---

* The `Baseline.ipynb` contains a baseline score generated using hardcoded options sequence and options policy