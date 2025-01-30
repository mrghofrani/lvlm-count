# LVLM-COUNT: Enhancing the Counting Ability of Large Vision-Language Models

# Installation

1. Setup conda environment.

```
conda create -n lvlmcount python=3.10.14
conda activate lvlmcount
```

2. Setup GroundingDino using the [link](https://github.com/IDEA-Research/GroundingDINO).
3. Download groundingdino weights.

```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

4. install the requirements.

```
pip install -r requirements.txt
```

5. Put the API keys inside the `.env` file in the following manner:

```
OPENAI_API_KEY=<Your GPT4o API key>
```
# How to run the project

## FSC-147

Use the `eval_fsc147.bash` script. Data are in `data/fsc-147/test.json` file. Image dataset is in [link](https://github.com/cvlab-stonybrook/LearningToCountEverything?tab=readme-ov-file#dataset-download).

```
bash eval_fsc147.bash
```

## TallyQA Simple Benchmark

Use the `eval_tallyqa_simplebenchmark` script. Data are in `data/tallyqa/benchmark_simple.json` file. Image dataset is in [link](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).

```
bash eval_tallyqa_simplebenchmark
```

## TallyQA Complex Benchmark

Use the `eval_tallyqa_complexbenchmark` script. Data are in `data/tallyqa/benchmark_complex.json` file. Image dataset is in [link](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).

```
bash eval_tallyqa_complexbenchmark
```

## Emoji-Bench

Use the `eval_emoji.bash` script. Data are in `emoji_benchmark/benchmark_one_canvas` directory.

```
bash eval_emoji.bash
```

## PASCAL VOC Benchmark

Use the `eval_pascal.bash` script. Data are in `data/pascal/sampled_pascal.csv` file. Image dataset is in [link](http://host.robots.ox.ac.uk/pascal/VOC/).

```
bash eval_pascal.bash
```

**Note:** In our project, just for the area detection and segmentation we used the code from [GrounddedSAM Project](https://github.com/IDEA-Research/Grounded-Segment-Anything).