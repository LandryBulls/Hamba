<div align="center">

# Hamba: Single-view 3D Hand Reconstruction with <br>Graph-guided Bi-Scanning Mamba
<div>
    <a href='https://www.haoyed.com/' target='_blank'>Haoye Dong*</a>&emsp;
    <a href='https://aviralchharia.github.io/' target='_blank'>Aviral Chharia*</a>&emsp;
    <a href='https://www.linkedin.com/in/wenbogou' target='_blank'>Wenbo Gou*</a>&emsp;</br>
    <a href='https://scholar.google.com/citations?user=3elKp9wAAAAJ' target='_blank'>Francisco Vicente Carrasco</a>&emsp;
    <a href='https://www.cs.cmu.edu/~ftorre/' target='_blank'>Fernando De la Torre</a>&emsp;
</div>
<div>
    Carnegie Mellon University&emsp;<br>
    *Equal Contribution
</div>
<div>
    <b>NeurIPS 2024</b>
</div>
<br>

  <a href="https://arxiv.org/abs/2407.09646"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2407.09646-00ff00.svg"></a>
  <a href="https://humansensinglab.github.io/Hamba"><img alt="Webpage" src="https://img.shields.io/badge/Webpage-up-yellow"></a>
  [![GitHub Stars](https://img.shields.io/github/stars/humansensinglab/Hamba?style=social)](https://github.com/humansensinglab/Hamba)

---

<img src="docs/pics/Teaser.jpg" width="49%"/>
<img src="docs/pics/In-the-wild_Additional.jpg" width="32.5%"/>

<strong> Hamba can achieve an accurate and robust reconstruction of 3D Hands.</strong><br>
:open_book: For more visual results, go checkout our <a href="https://humansensinglab.github.io/Hamba/" target="_blank">project page</a>
</div>

---

## :rocket: **Updates**
- <i>More Coming Soon</i>
- 🔲 **Coming Soon!**: Hamba Hugging Face. <i>Run Hamba with a single click!</i>
- ✅ **Mar. 31, 2025**: We released the Hamba Inference Codes and Weights.
- ✅ **Jan. 17, 2025**: Hamba featured in TechXplore! Check out our [Blog](https://techxplore.com/news/2025-01-ai-human.html).
- ✅ **Sep. 25, 2024**: Hamba accepted at NeurIPS. Check out our Poster [here](https://neurips.cc/virtual/2024/poster/93574). See everyone at Vancouver!</i>
- ✅ **Jul. 16, 2024**: Hamba project page is now live.
- ✅ **Jul. 12, 2024**: We released the Hamba Paper on arXiv. Check the preprint!

## :open_book: **Abstract**

3D Hand reconstruction from a single RGB image is challenging due to the articulated motion, self-occlusion, and interaction with objects. Existing SOTA methods employ attention-based transformers to learn the 3D hand pose and shape, yet they do not fully achieve robust and accurate performance, primarily due to inefficiently modeling spatial relations between joints. To address this problem, we propose a novel graph-guided Mamba framework, named Hamba, which bridges graph learning and state space modeling. Our core idea is to reformulate Mamba's scanning into graph-guided bidirectional scanning for 3D reconstruction using a few effective tokens. This enables us to efficiently learn the spatial relationships between joints for improving reconstruction performance. Specifically, we design a Graph-guided State Space (GSS) block that learns the graph-structured relations and spatial sequences of joints and uses 88.5% fewer tokens than attention-based methods. Additionally, we integrate the state space features and the global features using a fusion module. By utilizing the GSS block and the fusion module, Hamba effectively leverages the graph-guided state space features and jointly considers global and local features to improve performance. Experiments on several benchmarks and in-the-wild tests demonstrate that Hamba significantly outperforms existing SOTAs, achieving the PA-MPVPE of 5.3mm and F@15mm of 0.992 on FreiHAND. At the time of this paper's acceptance, Hamba holds the top position, <strong>Ranked 1</strong> in two Competition Leaderboards on 3D hand reconstruction.

## Installation

First you need to clone the repo:
```bash
git clone --recursive https://github.com/humansensinglab/Hamba.git
cd Hamba
```

We recommend creating a conda env for Hamba. 
```bash
conda create --name hamba python=3.10
conda activate hamba
```

Next, you can install other dependencies similar to [Hamer](https://github.com/geopavlakos/hamer/). Below is a sample for CUDA 11.7, but you can adapt accordingly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```

Next, you need to install dependencies for [State Space Model](https://github.com/state-spaces/mamba) and [VMamba](https://github.com/MzeroMiko/VMamba) repositories.

```bash
git clone https://github.com/state-spaces/mamba 
pip install causal-conv1d>=1.4.0
pip uninstall mamba-ssm
pip install setuptools==60.2.0
pip install .
```

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
cd kernels/selective_scan && pip install .
```

Download the MANO hand model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section. Our model only requires the right hand MANO model. Place the `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.


## Demo 
For demo, download the trained Hamba model weights [here](https://drive.google.com/file/d/1JRPC11YfQym8t_EZkhsroglvGHrGPbU-/view?usp=sharing).
```bash
python demo.py --checkpoint ckpts/hamba/checkpoints/hamba.ckpt --img_folder example_data --out_folder ./demo_out/example_data/ --full_frame
```


## Evaluation

We followed [MeshGraphormer](https://github.com/microsoft/MeshGraphormer) for Test-time augmentation and without Test-time augmentation evaluation on the FreiHAND benchmark. First download the [FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) dataset following the official repository. Download the trained Hamba model weights [here](https://drive.google.com/file/d/1Otp1Dbs9sekc08wTM3EI3q3wq4U4eVo6/view?usp=sharing).

### Without Test-Time Augmentation

Step 1: Compute the Results
```bash
python eval/test_on_augmentation/run_hand_multiscale.py --base_path eval_data/FreiHAND/FreiHAND_pub_v2_eval --checkpoint ckpts/hamba_freihand/checkpoints/hamba_freihand.ckpt --output_dir eval/freihand/results_woTTA/hamba_freihand/
```

Step 2: Find the Evaluation Metrics
```bash
python eval/freihand/eval_freihand_official.py \
--gt_dir eval_data/FreiHAND/FreiHAND_pub_v2_eval \
--output_dir eval/freihand/results_woTTA/hamba_freihand/ \
--pred_dir eval/freihand/results_woTTA/hamba_freihand/ \
--pred_file_name hamba-ckptxx-sc10_rot0-pred.zip
```

### With Test-Time Augmentation

Step 1: Compute the Results
```bash
python eval/test_on_augmentation/run_hand_multiscale.py --base_path eval_data/FreiHAND/FreiHAND_pub_v2_eval --checkpoint ckpts/hamba_freihand/checkpoints/hamba_freihand.ckpt --output_dir eval/freihand/results_TTA/hamba_freihand/ --multiscale_inference
```

Step 2: Find the Evaluation Metrics
```bash
python eval/freihand/eval_freihand_official.py \
--gt_dir eval_data/FreiHAND/FreiHAND_pub_v2_eval \
--output_dir eval/freihand/results_TTA/hamba_freihand/ \
--pred_dir eval/freihand/results_TTA/hamba_freihand/ \
--pred_file_name hamba-ckptxx-sc-rot-final-pred.zip
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=humansensinglab/Hamba&type=Date)](https://www.star-history.com/#humansensinglab/Hamba&Date)

## License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc]. 
Permission is granted for non-commercial research. For commerical use, please reachout to our Lab.

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Acknowledgements

Parts of the codes have been taken and adapted from the below repos. Please acknowledge and adhere to the licenses of each repository that Hamba builds upon.
- [Mamba](https://github.com/state-spaces/mamba)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [Hamer](https://github.com/geopavlakos/hamer/)
- [HMR](https://github.com/akanazawa/hmr)
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SemGCN](https://github.com/garyzhao/SemGCN)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)

## :bookmark_tabs: Citation
If you find our work useful for your project, please consider adding a star to this repo and citing our paper:
```bibtex
    @article{dong2024hamba,
      title={Hamba: Single-view 3d hand reconstruction with graph-guided bi-scanning mamba},
      author={Dong, Haoye and Chharia, Aviral and Gou, Wenbo and Vicente Carrasco, Francisco and De la Torre, Fernando D},
      journal={Advances in Neural Information Processing Systems},
      volume={37},
      pages={2127--2160},
      year={2024}
    }
```
