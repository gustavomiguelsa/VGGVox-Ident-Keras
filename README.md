# VGGVox-Keras
A literal translation of the MatConvNet VGGVox model for speaker identification to Keras.

I do not own the architecture of the VGGVox model, nor did I create it. The model and corresponing demo were developed by Arsha Nagrani and her colleagues, and are available at: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

This is a line-by-line transliteration of the MatLab MatConvNet implementation of the model to Python's Keras. I made this because there really is not a trustworthy and reliable translator for NNs from MatConvNet to Keras or other Python based frameworks. Even the caffe converter provided by MatConvNet for SimpleNNs does not work as advertised. Hence my own handmade translation. The demo is 100% working and yields the same results as the MatConvNet based demo.

Feel free to use this Keras implementation for your own research purposes, or the other signal processing scripts provided. If you run into any issues, feel free to contact me.

Cheers!

NOTE: The weights H5 file was compressed as a .zip due to GitHub's limit on large files over 100MB. Just extract the file using your usual extractor.
