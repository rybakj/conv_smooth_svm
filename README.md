# Convolution-smoothed SVM

This repo replicates simulations for convolutin-smooth SVM.

- The notebook `simulations.ipynb` generates the figures used to compare the convergence of Bahadur remainder and the Type 1 error rates of convolution-smoothed SVM vs the (hinge-loss) SVM.
- Figures are stored in `./outputs/figures` directory. 
- Data, generated by simulations, which underpin these figures are stored in `/outputs` directory.
- The implementation of convolution-smoothed SVM and of unregularised SVM can be found in `/src/models.py` file.
