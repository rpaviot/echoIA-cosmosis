# echoIA-cosmosis

Code used to fit wg+ and wgg with Cosmosis.
The measure folder show how to build a random catalog and compute wg+ for a galaxy sample.  It then converted the measurements in the .ascii format
into the .fits format convention of Cosmosis.

The theory folder show how to use harry pipeline https://github.com/harrysjohnston/2ptPipeline 
The cosmosis folder show the modification done on the cosmosis/cosmosis-standard-library modules to perform the fit.

What need to be done :
- Look at the impact of pimax and dpi on the measurement of wg+.
- Replace the Dc(z) measurement provided by pyccl in harry code directly by the camb calculation of Dc(z) (already computed within the camb module).
- Provide an other .ini file with the TATT model.
- Look more closely to harry' modules. Actually it doesn't go from Pk to Cl, weird convention for rp? (rp = theta here..)
- 
