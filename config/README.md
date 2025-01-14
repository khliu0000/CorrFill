# Configuration files for reproduction


* ```baseline```: Can be ```"ipadapterplus"```, ```"paintbyexample"```, ```"sidebyside"```, or ```"leftrefill"```.
* ```val_set```: Can be ```"realestate"``` or ```"megadepth"```.
* ```dominant_filter```: Switch for the dominant filter, a refining mechanism.
* ```smoothing```: Switch for smoothing, a refining mechanism.
* ```attn_masking```: Switch for attention masking, a guidance mechanism.
* ```latent_optimiza```: Switch for latent optimization, a guidance mechanism.
* ```str_masking```: The strength of the attention masking guidance.
* ```str_optimize```: The strength of the latent optimization guidance.
* ```random_seed```: Random seed for reproduction.
* ```step_masking```: Number of steps involving attention masking.
* ```step_optimize```: Number of steps involving latent optimization.
* ```mask_window```: Window size of attention masking. The size is the number of tokens if the value is positive, while the size represented by ```-mask_window``` is the ratio with respect to the image size.
* ```mask_boost```: A positive value added to the corresponding areas in the attention masking mechanism.
* ```smooth_window```: Window size of the smoothing. The value is the ratio to the image size.
* ```vote_blocks```: The set of blocks in the UNet to collect attention scores from.