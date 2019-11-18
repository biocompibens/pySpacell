# pySpacell
A Python package for single-cell spatial image analysis.

This toolbox allows to test for the presence of a spatial effect in microscopy images of adherent cells and estimate the spatial scale of this effect. It can be used for any light microscopy images of cells as well as other types of single-cell data such as in situ transcriptomics or metabolomics. Input format of our package matches standard output formats from image analysis tools such as *CellProfiler*, *Fiji* or *Icy*, and thus makes our toolbox easy and straightforward to use, yet offering a powerful statistical approach for a wide range of applications.

The available spatial tests are:
* for the continuous variables
    * **auto-correlation indices (Moran, Gety's, Geary)** both **global (per image)** and **local (per cell)**. This package relies on the [`PySAL`](https://pysal.readthedocs.io/en/latest/) python package implementation 
* for the categorical variables.
    * **Newman's assortativity**: it addresses the question of whether there are **more self-connections between objects of the same category than expected at random**. (cf Newman, M. E. J. (2003). Mixing patterns in networks. Physical Review E - Statistical Physics, Plasmas, Fluids, and Related Interdisciplinary Topics, 67(2), 13. https://doi.org/10.1103/PhysRevE.67.026126; see [pdf](http://www.uvm.edu/pdodds/research/papers/others/2003/newman2003e.pdf))
    * **Ripley's K and L functions**: it explores both the spatial layout of your objects (K or L function) and the spatial layout of pairs of categories one compared to the other (K or L cross-functions). See PySpacell documentation and paper for more details. (cf Dixon, P. M. (2002). Ripley’s K function. Encyclopedia of Environmetrics, 3, 1796–1803. see [pdf](https://www3.nd.edu/~mhaenggi/ee87021/Dixon-K-Function.pdf))


## Availability
* It runs with python3.6
* It is available through `pip install pySpacell` or downloadable via `git clone https://github.com/biocompibens/pySpacell.git`
* NB: the git depot contains large data files. To download them correctly, `git lfs clone https://github.com/biocompibens/pySpacell.git`. The rest of the files are not impacted by a standard git clone.

## Usage examples
Example data are located in *data/* along with exemple scripts in *example/*

### Basic example

This example of code will compute a series of neighborhood matrices, calculate a series of spatial-autocorrelation Moran indices and plot it in a correlogram (line plot with x-axis corresponding to the increasing neighborhoods and y-axis the value of the Moran index).

```
import matplotlib.pyplot as plt
from pySpacell import Spacell

image_label_file = ## path to image label file
feature_file = ## path to csv file containing the features

column_x_y = ['X', 'Y']
column_object = 'label'

spa = Spacell(feature_file, 
         image_label_file, 
         column_x_y=column_x_y,
         column_objectnumber=column_object)
         
feature = ## corresponding to the name of the column in the csv
method = 'moran' ## or 'geary' or 'getisord' for global spatial-autocorrelation indices

neighborhood_matrix_type = 'k' ## or 'radius', or 'network'
neighborhood_p0 = 0 ## lower bound
neighborhood_p1 = 30 ## upper bound
neighborhood_step = 5 ## step to iterate between lower and upper bounds

permutations = 999 ## number of permutations to construct the null model
quantiles = [2.5, 97.5] ## quantiles for the null distribution and the statistical test

seq_points_x = spa.correlogram(feature,
                                method,
                                neighborhood_matrix_type,
                                neighborhood_p0,
                                neighborhood_p1,
                                neighborhood_step=neighborhood_step,
                                permutations=permutations,
                                quantiles=quantiles, 
                                plot_bool=True)

plt.show()

```

Main methods of the pySpacell class include:
* *correlogram*: computes a serie of spatial analysis tests for the provided features. Gives one value for the image. The starting and ending neighborhood parameters, neighborhood_p0 and neighborhood_p1, are to be set. 3 modes are available to define the intermediary neighborhood parameters.

* *get_neighborhood_matrix*: return the neighborhood matrix for specified parameters if already computed. 3 modes are available: 
    1. 'k' for k-nearest neighbors;
    2. 'radius' for neighbors at an euclidean distance;
    3. 'network' for neighbors from the object graph (touching objects are neighbors).
    
For each mode, an interval is requested to know which neighbors to include.
* *compute_per_image_analysis*: computes per image spatial analysis tests for the provided features for one neighborhood matrix.
* *compute_per_object_analysis*: computes per object spatial analysis tests for the provided features for one neighborhood matrix.

The two last methods can be used for both continuous and categorical variables, using the `method` argument to choose the type of analysis.

There are additional methods for visualization: *get_feature_filled_image*, *plot_2im*, *plot_neighborhood*, *plot_correlogram*, *plot_ripley_cross*, *plot_ripley_diff*, *get_hot_spots_image*.

### `example/script_figure_FUCCI.py`

* This script allows to reproduce results from the Figure 3 of the paper.
* The biological experiment (see paper for more details - Figure 3 - and interpretation):
    * MDCKII (Madin-Darby Canine Kidney Type II) cells were transfected with the fluorescence ubiquitination cell cycle indicator (FUCCI) system ([Thermofisher](https://www.thermofisher.com/fr/fr/home/life-science/cell-analysis/cell-viability-and-regulation/cell-cycle/live-cell-imaging-of-cell-cycle-and-division.html)). 
    * The FUCCI system allows to study the cell cycle progression. This system relies on two fused fluorescent proteins, Cdt1-RFP and geminin-GFP, the Cdt1-RFP (red) only expressing in the G1 phase, the geminin-GFP (green) in the S, G2, and M phases. During the G1/S transition, both proteins are present, hence nuclei appear yellow on an overlay image.    
    * Additionally to the FUCCI system, cells were stained with Hoechst labeling every nucleus.    
* **Asked biological question:** do the cells tend to be in the same cell cycle phase when they are in the vicinity of each other? The ratio of fluorescence between the green and the red is considered as a **continuous** variable. 
* It relies on `data/FUCCI_*` files.
* It provides an example how to:
    * load data;
    * create a pySpacell object;
    * compute neighborhood matrices;
    * compute auto-correlation index globaly (per image) and visualize it;
    * compute auto-correlation index locally (per cell) on a small region of the original image and visualize it.


### `example/script_visualize_neighbors.py`

* This script allows to visualize neighbor connections.
* It relies on `data/FUCCI_crop*` files.
* It shows how to:
    * load data;
    * create a pySpacell object;
    * compute one neighborhood matrix;
    * show the cell neighbor connections directly on the image space.
    
### `example/script_figure_co-culture.py`

* This script allows to reproduce the results from the figure 4 of the paper.
* Biological experiment:
    * Hela cells and NIH3T3 fibroblasts were co-cultured and seeded three days on a glass slide before imaging. HeLa and NIH3T3 cells constitutively expressed H2B-mCherry and GFP respectively. 
    * Cells were detected manually with [Cell Counter ImageJ plugin](https://imagej.net/Cell_Counter).
* **Asked biological question:** do the cells of one cell type (Hela cells or fibroblasts) tend to cluster with one's cell type? The cell type is here seen as a **categorical** variable.
* It relies on `data/co-culture*` files.
* It shows how to:
    * load data;
    * create a pySpacell object;
    * compute and display a correlogram, which is one global statistics computed at different scales, i.e. with different neighborhood matrices; it does so for both Newman's assortativity and Ripley's K functions on the cell type (categorical variable).
    * exemplify the use of pair distance constraints for the null model randomizations (see paper for more details).
    * display a single Ripley's cross-functions difference to visualize the spatial layout of a chosen pair of categories. 
    
    

# Citation

*PySpacell : a Python package for spatial analysis of cell images.* Rose F., Rappez L., Triana S. H., Alexandrov T., Genovesio A. [Under review]
