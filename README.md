# FaceToSimpson
FaceToSimpson is a Deep Neural Network that is able to transform a human face into a Simpson face, and viceversa. It's a re-implementaion of CycleGAN, trained on a dataset composed by two domains: X, a set of human faces, and Y, a set of Simpson faces. The test simply computes the FID between the Simpson faces in the test dataset, and the Simpson faces generated starting from the human faces of the test dataset. In order to compute the FID score, it was embedded the source code of the following repository: https://github.com/yhlleo/GAN-Metrics/

CycleGAN Paper: https://arxiv.org/pdf/1703.10593.pdf
