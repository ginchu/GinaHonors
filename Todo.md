<ul>
<li> Stop using notebooks. Move notebooks into some directory so we have a copy. Split code between Dataset creation and training.  
<li> Add option to data preparation: target is either energy, or atomization energy
<li> Split data train and test, write code for testing
<li> Experiment with removing nodes when they do not have features: https://docs.dgl.ai/en/0.6.x/generated/dgl.to_simple.html
<li> Add to the code invariance to rotation, translation, etc...
<li> Continue adding more information for wandb
<li> Add another representation for local environments, Coulom matrix (sorted):  https://singroup.github.io/dscribe/latest/tutorials/descriptors/coulomb_matrix.html
    <ul>
    <li> G: done but wouldn't there be duplicates of src, dest pairs and their distance in the graph if points share neighbors
    <li> G: also not sure of what the difference is with coulomb and what we did previously. since all distances calculated will be in one graph, wouldn't the previous method put the distance between two molecules eventually (ie in h2o wouldn't the dist betw the two H eventually be put in the graph)?
    </ul>
</ul>
