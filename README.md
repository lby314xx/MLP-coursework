# MLP-coursework
An implementation of SKNet to image super resolution

# MLP-SRSKNet


* Dataset:    ## file size too big using matlab,insufficient memory#
                   1. just use one downsize x4 smaller file size and faster  
                   2. use python instead-- opencv could do downscaling but our model need be modified a lot, like data loader, image dimension
                   
* Inplementation:      ## We use existing Super resolution Structure called VDSR, and replace the model by our SKnet.#
                   1. since we are using different network, things need to change like inputs, dimensions. 
* Improvement:    ## hyper-parameters & loss function & optimisation method#
* Comparisons:    ## need to compare the performance with other networks#
