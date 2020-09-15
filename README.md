CONVOLUTIONAL VIDEO CAPTIONING

# Video Captioning using CNN decoder 

## Abstract:
Temporal action localization is an important yet challenging problem. Given a long, untrimmed video consisting of multiple action instances and complex background contents, we need not only to recognize their action categories, but also to localize the start time and end time of each instance.Although Many state-of-the-art action localization models use 3D convnets (C3D network) which are efficient in abstracting action semantics achieve accurate recognition of actions but can not precisely localize  these actions as temporal length of the input reduces by factor of 8.So,to achieve precise localization of action boundaries this paper has designed a novel Convolutional-De-Convolutional(CDC) network,which places CDC layers on top of 3D convnets .These CDC layers perform temporal upsampling and spatial downsampling operations simultaneously to predicts action instances at frame level granularity .Finally,the CDC network demonstrates high efficiency in jointly modeling  action semantics in space-time and fine-grained tem
poral dynamics. 
## Architecture 
![alt text](https://drive.google.com/file/d/11HbiglG296qm08eSpVn8Vem9oXCHMD-K/view?usp=sharing)
