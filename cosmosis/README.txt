Here are the main change performed in order to do a fit of wg+ with Cosmosis.
All these python code can be found within cosmosis directory of the cosmosis env.

For me it is within the conda folder: miniforge3/envs/yourenv/lib/python3.9/site-packages/cosmosis/ 

The gaussian_likelihood can be found within that folder, and type_table.txt and twopoint.py
can be found in cosmosis-standard-library folder on the 2pt section.

In the twopoint.py is add two quantities in the class Type(enum):
galaxy_position_red = "GPS", and  galaxy_shear_plus_red = "G+S" to specify that we are
considering redshift space quantities (compare to standard real space quantities of Cosmosis.)

Then, we link these quantities (as denoted by QUANT1 and QUANT2 in the data fits file) to a given Cosmosis section, as given by these two extra lines in the type_table.txt:

galaxy_position_red	    galaxy_shear_plus_red	 projected_galaxy_intrinsic	theta	bin_{0}_{1}
galaxy_position_red	    galaxy_position_red 	 projected_galaxy_power		theta 	bin_{0}_{1}


Finally, Cosmosis doesn't understand just yet that we want to fit these extra quantities.
Therefore, two extra function are add to the gaussian_likelihood code;
extract_theory_points_wgp and extract_theory_points_wgg, and this functions are used instead of the generic function extract_theory_points at the beginning of the function do_likelihood.


