# jpeg-compression-art

This repo illustrates the full process of jpeg compression, going in depth to clarify the discrete cosine transforms and quantization steps. 

Furthermore, it yields artistic results by compressing a single channel of the image.


how to use~

cd src 

python jpeg_compression <path_to_image> <size_of_block> <quant_value>

generally, 

7 < size_of_block < 15 

20 < quant_value < 100 

but it's interesting to experiment with these values. 
