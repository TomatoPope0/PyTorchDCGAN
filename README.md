# PyTorch Implementation of Deep Convolutional Generative Adversarial Networks (DCGAN)

This is my implementation of DCGAN (Radford & Metz, 2016) in PyTorch (Pazkea et al., 2019).

As I'm an active learning student, this implementation may not be complete or accurate. Therefore, I recommend you to use other reliable implementations if you're willing to use it in your project.

If you find any bugs or flaws, please let me know. Also, I tried using the APA style citation, just for practicing. If you think any of the citations are improper, please let me know.

## Model Outputs

Training on other datasets (like LSUN (Yu, Zhang, Song, Seff, & Xiao, 2015) or CelebA (Liu, Luo, Wang, &Tang, 2015)) or on higher epochs will be performed later.

### Single epoch on MNIST (LeCun, Bottou, Bengio, & Haffner, 1998)

Training MNIST (LeCun et al., 1998) for higher epoch causes generator to collapse - generating only two images. I guess this is because of the small dataset.

![](./Outputs/G-1/image0.bmp)
![](./Outputs/G-1/image1.bmp)
![](./Outputs/G-1/image2.bmp)
![](./Outputs/G-1/image3.bmp)
![](./Outputs/G-1/image4.bmp)
![](./Outputs/G-1/image5.bmp)
![](./Outputs/G-1/image6.bmp)
![](./Outputs/G-1/image7.bmp)
![](./Outputs/G-1/image8.bmp)
![](./Outputs/G-1/image9.bmp)

### 10 epochs on CelebA (Liu et al., 2015)

The model trained with the 64x64 sized cetercrop of aligned images.

![](./Outputs/G-10-Face/image0.bmp)
![](./Outputs/G-10-Face/image1.bmp)
![](./Outputs/G-10-Face/image2.bmp)
![](./Outputs/G-10-Face/image3.bmp)
![](./Outputs/G-10-Face/image4.bmp)
![](./Outputs/G-10-Face/image5.bmp)
![](./Outputs/G-10-Face/image6.bmp)
![](./Outputs/G-10-Face/image7.bmp)
![](./Outputs/G-10-Face/image8.bmp)
![](./Outputs/G-10-Face/image9.bmp)

## Pretrained Models

You can download pretrained generators `G-n.pt` from the `Models` folder. `n` is the number of epochs used for training.

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278-2324.
- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep Learning Face Attributes in the Wild. *Proceedings of International Conference on Computer Vision*.
- Pazke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems 32* (pp. 8024-8035). Curran Associates, Inc. 
- Radford, A., & Metz, L. (2016). *Unsupervised Representational Learning with Deep Convolutional Generative Adversarial Networks*. *arXiv preprint*. arXiv:1511.06434v2
- Yu, F., Zhang, Y., Song, S., Seff, A. & Xiao, J. (2015). LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop. *arXiv preprint*. arXiv:1506:03365