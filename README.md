# Official Implementation of h/BehaveMAE and Shot7M2 (ECCV 2024)

> **Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders**<br>
> [Lucas Stoffl](https://people.epfl.ch/lucas.stoffl?lang=en), [Andy Bonnetto](https://people.epfl.ch/andy.bonnetto?lang=en), [Stéphane d'Ascoli](https://sdascoli.github.io/), [Alexander Mathis](https://www.mathislab.org/)<br>École Polytechnique Fédérale de Lausanne (EPFL)
>
> [[ECCV'24](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8076_ECCV_2024_paper.php)], [[bioRxiv](https://www.biorxiv.org/content/10.1101/2024.08.06.606796v1)]


<img src="media/overview.png" width="600">


## 📰 News
**[2024.10]**  We released the code and datasets for h/BehaveMAE, Shot7M2 and hBABEL🎈<br>
**[2024.07]**  This work is accepted to ECCV 2024 🎉 -- see you in Milano!<br>
**[2024.06]**  h/BehaveMAE and Shot7M2 were presented at [FENS Forum 2024](https://fensforum.org/)<br>


## 🎥 Teaser Presentation
<a href="https://www.youtube.com/watch?v=J8d2jxc5UNo">
  <img src="https://img.youtube.com/vi/J8d2jxc5UNo/0.jpg" alt="Teaser Presentation" width="400"/>
</a>


## ✨ Highlights

### 🔥 Hierarchical Action Segmentation (HAS) Benchmarks
Recognizing the scarcity of large-scale hierarchical behavioral benchmarks, we create a novel synthetic basketball playing benchmark (Shot7M2). Beyond synthetic data, we extend BABEL into a hierarchical action segmentation benchmark (hBABEL).

### ⚡️ A Generalized and Hierarchical Masked Autoencoder Framework

We developed a masked autoencoder framework (hBehaveMAE) to elucidate the hierarchical nature of motion capture data in an unsupervised fashion. We find that hBehaveMAE learns interpretable latents, where lower encoder levels show a superior ability to represent fine-grained movements, while higher encoder levels capture complex actions and activities.


## 🔨 Installation

We developed and tested our models with `python=3.9.15`, `pytorch=2.0.1`, and `cuda=11.7`. Other versions may also be suitable.
The easiest way to set up the environment is by using the provided `environment.yml` file:

  ```bash
  conda env create -f environment.yml
  conda activate behavemae
  ```


The evaluation scripts also rely on `GNU Parallel` for faster processing. If you encounter a `parallel: command not found` error, please install it using your system's package manager. For example, on Debian/Ubuntu:
```bash
sudo apt-get update && sudo apt-get install parallel
```

## 🔧 Troubleshooting

*   **`AttributeError: module 'torch.optim' has no attribute '_multi_tensor'`**: This error may occur with recent versions of PyTorch where internal APIs have changed. The fix is applied in `main_pretrain.py` by changing `torch.optim._multi_tensor.AdamW` to the public `torch.optim.AdamW`. If you encounter this, ensure your code is up-to-date with the repository.

*   **`CUDA out of memory` on consumer GPUs (e.g., RTX 3060)**: The default `batch_size` of 512 in the training scripts (e.g., `scripts/shot7m2/train_hBehaveMAE.sh`) is configured for high-end GPUs (like A100 or V100). To run pre-training on GPUs with less VRAM (e.g., 12GB), you need to adjust the batch size and use gradient accumulation to maintain the effective batch size.

    For example, in `scripts/shot7m2/train_hBehaveMAE.sh`, we have modified the parameters as follows:
    -   `--batch_size` changed from `512` to `64`.
    -   `--accum_iter 8` was added.

    This keeps the effective batch size at `64 * 8 = 512` while significantly reducing memory consumption. You may need to tune these values depending on your specific GPU.

## ➡️ Data Preparation

For downloading and preparing the three benchmarks Shot7M2 ([download here 🏀](https://huggingface.co/datasets/amathislab/SHOT7M2)), hBABEL, and MABe22 we compiled detailed instructions in the [datasets README](datasets/README.md).


## 🔄 Pre-training

To pre-train a model on 2 GPUs:
```
bash scripts/shot7m2/train_hBehaveMAE.sh 2
```

## ⤴️ Inference
To extract hierarchical embeddings after training and evaluate these embeddings:
```
bash scripts/shot7m2/test_hBehaveMAE.sh
```

## 🦁 Model Zoo

We provide a collection of pre-trained models on [zenodo](https://zenodo.org/records/13790191) that were reported in our paper, allowing you to reproduce our results:

|  Method  | Dataset | Checkpoint |
| :------: | :-----: | :--------: |
| hBehaveMAE | Shot7M2 | [checkpoint](https://zenodo.org/records/13790191/files/hBehaveMAE_Shot7M2.pth?download=1) |
| hBehaveMAE | hBABEL | [checkpoint](https://zenodo.org/records/13790191/files/hBehaveMAE_hBABEL.pth?download=1) |
| hBehaveMAE | MABe22 | [checkpoint](https://zenodo.org/records/13790191/files/hBehaveMAE_MABe22.pth?download=1) |


## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

#### 📄 bioRxiv Version
```
@article{stoffl2024elucidating,
  title={Elucidating the Hierarchical Nature of Behavior with Masked Autoencoders},
  author={Stoffl, Lucas and Bonnetto, Andy and d'Ascoli, Stephane and Mathis, Alexander},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
#### 📘 ECCV Version
```
@inproceedings{stoffl2024elucidating,
  title={Elucidating the hierarchical nature of behavior with masked autoencoders},
  author={Stoffl, Lucas and Bonnetto, Andy and d’Ascoli, St{\'e}phane and Mathis, Alexander},
  booktitle={European Conference on Computer Vision},
  pages={106--125},
  year={2024},
  organization={Springer}
}
```

## 👍 Acknowledgements

We thank the authors of the following repositories for their amazing work, on which part of our code is based:
- **[Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://github.com/facebookresearch/hiera)**
- **[Evaluator code for MABe 2022 Challenge](https://github.com/damaggu/MABe2022)**

## 🔒 Licensing

This repository is licensed under two different licenses depending on the codebase:

- **Apache 2.0 License**: The majority of the project, including all original code and modifications.
- **CC BY-NC 4.0 License**: The code inside the `hierAS-eval/` directory is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. This means it cannot be used for commercial purposes.

Please refer to the respective `LICENSE` file in the root of the repository and in `hierAS-eval/` for more details.
