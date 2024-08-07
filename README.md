# SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations

![SDEdit](images/sde_animation.gif)

[**Project**](https://sde-image-editing.github.io/) | [**Paper**](https://arxiv.org/abs/2108.01073) | [**Colab**](https://colab.research.google.com/drive/1KkLS53PndXKQpPlS1iK-k1nRQYmlb4aO?usp=sharing)

PyTorch implementation of **SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations** (ICLR 2022).

[Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Yutong He](http://web.stanford.edu/~kellyyhe/), [Yang Song](https://yang-song.github.io/), [Jiaming Song](http://tsong.me/),
[Jiajun Wu](https://jiajunwu.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Stefano Ermon](https://cs.stanford.edu/~ermon/)

Stanford and CMU

![teaser](images/teaser.jpg)

Recently, SDEdit has also been applied to text-guided image editing with large-scale text-to-image models. Notable examples include [Stable Diffusion]("https://en.wikipedia.org/wiki/Stable_Diffusion) img2img function (see  [here]("https://github.com/CompVis/stable-diffusion#image-modification-with-stable-diffusion")), [GLIDE](https://arxiv.org/abs/2112.10741), and [distilled-SD](https://arxiv.org/abs/2210.03142). The below example comes from [distilled-SD](https://arxiv.org/abs/2210.03142).

![text_guided_img2img](images/text_guided_img2img.png)

## Overview

The key intuition of SDEdit is to "hijack" the reverse stochastic process of SDE-based generative models, as illustrated in the figure below. Given an input image for editing, such as a stroke painting or an image with color strokes, we can add a suitable amount of noise to make its artifacts undetectable, while still preserving the overall structure of the image. We then initialize the reverse SDE with this noisy input, and simulate the reverse process to obtain a denoised image of high quality. The final output is realistic while resembling the overall image structure of the input.

![sde_stroke_generation](images/sde_stroke_generation.jpg)

## Getting Started

The code will automatically download pretrained SDE (VP) PyTorch models on
[CelebA-HQ](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt),
[LSUN bedroom](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt),
and [LSUN church outdoor](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt).

### Data format

We save the image and the corresponding mask in an array format ``[image, mask]``, where
"image" is the image with range ``[0,1]`` in the PyTorch tensor format, "mask" is the corresponding binary mask (also the PyTorch tensor format) specifying the editing region.
We provide a few examples, and ``functions/process_data.py``  will automatically download the examples to the ``colab_demo`` folder.

### Re-training the model

Here is the [PyTorch implementation](https://github.com/ermongroup/ddim) for training the model.

## Stroke-based image generation

Given an input stroke painting, our goal is to generate a realistic image that shares the same structure as the input painting.
SDEdit can synthesize multiple diverse outputs for each input on LSUN bedroom, LSUN church and CelebA-HQ datasets.

To generate results on LSUN datasets, please run

```shell
python main.py --exp ./runs/ --config configs/bedroom.yml --sample -i images --mask_image_file lsun_bedroom1 --sample_step 3 --t 500  --ni
```

```shell
python main.py --exp ./runs/ --config configs/church.yml --sample -i images --mask_image_file lsun_church --sample_step 3 --t 500  --ni
```

![stroke_based_generation](images/stroke_based_generation.jpg)

## Stroke-based image editing

Given an input image with user strokes, we want to manipulate a natural input image based on the user's edit.
SDEdit can generate image edits that are both realistic and faithful (to the user edit), while avoid introducing undesired changes.

![stroke_edit](images/stroke_edit.jpg)

To perform stroke-based image editing, run

```shell
python main.py --exp ./runs/  --config configs/church.yml --sample -i images --mask_image_file lsun_edit --sample_step 3 --t 500  --ni
```

## Additional results

![stroke_generation_extra](images/stroke_generation_extra.jpg)

## References

If you find this repository useful for your research, please cite the following work.

```bibtex
@inproceedings{
      meng2022sdedit,
      title={{SDE}dit: Guided Image Synthesis and Editing with Stochastic Differential Equations},
      author={Chenlin Meng and Yutong He and Yang Song and Jiaming Song and Jiajun Wu and Jun-Yan Zhu and Stefano Ermon},
      booktitle={International Conference on Learning Representations},
      year={2022},
}
```

This implementation is based on / inspired by:

- [DDIM PyTorch repo](https://github.com/ermongroup/ddim).
- [DDPM TensorFlow repo](https://github.com/hojonathanho/diffusion).
- [PyTorch helper that loads the DDPM model](https://github.com/pesser/pytorch_diffusion).
- [code structure](https://github.com/ermongroup/ncsnv2).

Here are also some of the interesting follow-up works of SDEdit:

- [Image Modification with Stable Diffusion](https://github.com/CompVis/stable-diffusion#image-modification-with-stable-diffusion)
