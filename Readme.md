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

5. Create and `.env` file in the base project directory, and put the API keys in the file with the following manner:

```
OPENAI_API_KEY=<Your GPT4o API key>
```
# How to run the project

## FSC-147

Download the images of the dataset from [link](https://github.com/cvlab-stonybrook/LearningToCountEverything?tab=readme-ov-file#dataset-download). Place the images at `data/fsc-147`. The folder should look like

```
data
└─ 📂fsc-147/
    └─ 📜 test.json
    └─ 📂 images_384_VarV2
```

After setting up the data please run:

```
bash eval_fsc147.bash
```

## TallyQA Simple Benchmark

Download the images of the dataset from [link](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html). Place the images at `data/tallyqa/`. The folder should look like

```
data
└─ 📂 tallyqa/
    └─ 📜 benchmark_simple.json
    └─ 📜 benchmark_simple.json
    └─ 📂 genome
```

After setting up the data please run:

```
bash eval_tallyqa_simplebenchmark
```

## TallyQA Complex Benchmark

Download the images of the dataset from [link](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html). Place the images at `data/tallyqa/`. The folder should look like

```
data
└─ 📂 tallyqa/
    └─ 📜 benchmark_simple.json
    └─ 📜 benchmark_complex.json
    └─ 📂 genome
```

After setting up the data please run:

```
bash eval_tallyqa_complexbenchmark
```

## Emoji-Bench

The entire dataset is already in `emoji_benchmark/benchmark_one_canvas`. Please run the command:

```
bash eval_emoji.bash
```

## PASCAL VOC Benchmark

Download the images of the dataset from [link](http://host.robots.ox.ac.uk/pascal/VOC/). Place the images at `data/pascal/`. The folder should look like

```
data
└─ 📂 tallyqa/
    └─ 📜 sampled_pascal.csv
    └─ 📂 VOCdevkit
```

After setting up the data please run:

```
bash eval_pascal.bash
```

**Note:** In our project, just for the area detection and segmentation we used the code from [GrounddedSAM Project](https://github.com/IDEA-Research/Grounded-Segment-Anything).