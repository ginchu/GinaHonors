**DUNN FOLDER**
- bilayer, diamond, graphite, and monolayer py files pick 25 xyz files from each folder to create a 100 sample dataset
	- these files also create target_e.txt that lists the given energies of the 100 samples and quip_e.txt that lists the energies of the 100 samples calculated by quip 
- filenames.txt has the file names of the 100 sample dataset used so that it can be used as a test dataset
- mse.py finds the mse of the 100 sample dataset using target_e.txt and quip_e.txt
	- run this file to print out the mse