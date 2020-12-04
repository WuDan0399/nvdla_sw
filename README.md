# nvdla_sw


Matched result for:

    - Trained Lenet model compiled by native NVDLA compiler, with last layer removed (softmax, or get all 0);
 
    - Relay IR of Lenet with same trained weights compiled by modified NVDLA compiler, last layer removed.

Unmatched result for:

    - Output of Lenet with random weights, running in TVM;
    
    - Relay IR of Lenet with same random weights compiled by modified NVDLA compiler.
    
Reasons for unmatched results in TVM and NVDLA:

    - NVDLA only supports 1 byte int and 2 bytes half-precision float when dump into loadable file.
    
    - To fully utilize these bytes, it scales weights.
    
    
Scripts for extract caffe model weights: https://github.com/nilboy/extract-caffe-params


Other Notes:

  I load weights into float from the json file, but originally they should be double. I haven't changed it yet as the loadable file only supports half precision or int. No matter what I use there must bu some loss.
