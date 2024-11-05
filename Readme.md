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

4. Setup RealSRGAN using the [link](https://github.com/xinntao/Real-ESRGAN)
5. Download super resolution weights.

```
cd Real_ESRGAN
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

6. install the requirements.

```
pip install -r requirements.txt
```

7. Put the API keys inside the `.env` file in the following manner:

```
OPENAI_API_KEY=<Your GPT4o API key>
GEMINI_API_KEY=<Your Gemini API key>
```
# How to run the project

## FSC-147

Use the `eval_fsc147.bash` script.

```
bash eval_fsc147.bash
```

## TallyQA Simple Benchmark

Use the `eval_tallyqa_simplebenchmark` script.

```
bash eval_tallyqa_simplebenchmark
```

## TallyQA Complex Benchmark

Use the `eval_tallyqa_complexbenchmark` script.

```
bash eval_tallyqa_complexbenchmark
```

## Emoji-Bench

Use the `eval_emoji.bash` script.

```
bash eval_emoji.bash
```

**Note:** In our project, just for the area detection and segmentation we used the code from [GrounddedSAM Project](https://github.com/IDEA-Research/Grounded-Segment-Anything).