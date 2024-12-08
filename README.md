# VideoQA-TA
Multi-modal video question-answering
<p align="left">
    <img src="https://github.com/ZhixuanWuA/VideoQA-TA/blob/main/ICONS.png" width="150" style="margin-bottom: 0.2;"/>
<p>


## 📝 Change Logs
* [2024-09-09]: Create the Projects, and upload data, pre-trained LLM, ablation study links.
* All code will be released after the work is accepted.

## 🛠️ Requirements and Installation
Basic Dependencies:
* Python >= 3.8
* Pytorch >= 1.10.0
* CUDA Version >= 11.8
* transformers >= 4.28.0

**[Online Mode]** Install required packages (better for development):
```bash
git clone https://github.com/ZhixuanWuA/VideoQA-TA
cd VideoQA-TA
pip install -r requirements.txt
```

**[Offline Mode]** Install VideoQA-TA as a Python package (better for direct use):
```bash
git clone https://github.com/ZhixuanWuA/VideoQA-TA
cd VideoQA-TA
pip install --upgrade pip  
pip install -e .
```

## 📊 Datasets
We conduct experiments on [MSVD-QA](https://github.com/xudejing/video-question-answering) , [MSRVTT-QA](https://github.com/xudejing/video-question-answering), and [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) datasets. 
Then extract video frames from each video at 10 fps (https://github.com/boheumd/MA-LMM/blob/main/data/extract_frames.py), based on the annotations of each dataset, object detection with pre-trained RAM++ (https://github.com/IDEA-Research/Grounded-Segment-Anything).
Suppose your data structure is like:
```bash
datasets
│   ├── MSVD
│   |   ├── frames
│   |   ├── videos
|   |   └── annoations
|   |   └── objectdetected
...
```


## 🗝️ Training & Evaluation
**[Pre-trained LLM]** 
We use the pre-trained Q-Former weights from [InstructBLIP](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) and [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) as the LLM.

**[Training]** 
```bash
bash VideoQA-TA/scripts/${dataset_name}/train.sh
```

**[Testing]** 
```bash
bash VideoQA-TA/scripts/${dataset_name}/test.sh ${checkpoint_path}
```

**[Comparison with state-of-the-art methods]** 
1) [mPLUG-2](https://github.com/X-PLUG/mPLUG-2)
```bibtex
@inproceedings{xu2023mplug,
  title={mplug-2: A modularized multi-modal foundation model across text, image and video},
  author={Xu, Haiyang and Ye, Qinghao and Yan, Ming and Shi, Yaya and Ye, Jiabo and Xu, Yuanhong and Li, Chenliang and Bi, Bin and Qian, Qi and Wang, Wei and others},
  booktitle={International Conference on Machine Learning},
  pages={38728--38748},
  year={2023}
}
```
2) [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
```bibtex
@article{zhang2023video,
  title={Video-llama: An instruction-tuned audio-visual language model for video understanding},
  author={Zhang, Hang and Li, Xin and Bing, Lidong},
  journal={arXiv preprint arXiv:2306.02858},
  year={2023}
}
```
3) [Video-LLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
```bibtex
@article{cheng2024videollama,
  title={VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs},
  author={Cheng, Zesen and Leng, Sicong and Zhang, Hang and Xin, Yifei and Li, Xin and Chen, Guanzheng and Zhu, Yongxin and Zhang, Wenqi and Luo, Ziyang and Zhao, Deli and others},
  journal={arXiv preprint arXiv:2406.07476},
  year={2024}
}
```
4) [VideoChat](https://github.com/OpenGVLab/Ask-Anything/tree/main)
```bibtex
@article{2023videochat,
  title={VideoChat: Chat-Centric Video Understanding},
  author={KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao},
  journal={arXiv preprint arXiv:2305.06355},
  year={2023}
}
```
5) [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main)
```bibtex
@inproceedings{li2024mvbench,
  title={Mvbench: A comprehensive multi-modal video understanding benchmark},
  author={Li, Kunchang and Wang, Yali and He, Yinan and Li, Yizhuo and Wang, Yi and Liu, Yi and Wang, Zun and Xu, Jilan and Chen, Guo and Luo, Ping and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22195--22206},
  year={2024}
}
```
6) [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
```bibtex
@inproceedings{Maaz2023VideoChatGPT,
    title={Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models},
    author={Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Khan, Fahad Shahbaz},
    booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)},
    year={2024}
}
```
7) [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
```bibtex
@article{lin2023video,
  title={Video-llava: Learning united visual representation by alignment before projection},
  author={Lin, Bin and Zhu, Bin and Ye, Yang and Ning, Munan and Jin, Peng and Yuan, Li},
  journal={arXiv preprint arXiv:2311.10122},
  year={2023}
}
```
8) [Movie-LLM](https://github.com/Deaddawn/MovieLLM-code)
```bibtex
@article{song2024moviellm,
  title={Moviellm: Enhancing long video understanding with ai-generated movies},
  author={Song, Zhende and Wang, Chenchen and Sheng, Jiamu and Zhang, Chi and Yu, Gang and Fan, Jiayuan and Chen, Tao},
  journal={arXiv preprint arXiv:2403.01422},
  year={2024}
}
```
9) [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi/tree/main)
```bibtex
@article{jin2023chatunivi,
  title={Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding}, 
  author={Peng Jin and Ryuichi Takanobu and Caiwan Zhang and Xiaochun Cao and Li Yuan},
  journal={arXiv preprint arXiv:2311.08046},
  year={2023}
}
```
10) [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
```bibtex
@article{li2023llama,
  title={Llama-vid: An image is worth 2 tokens in large language models},
  author={Li, Yanwei and Wang, Chengyao and Jia, Jiaya},
  journal={arXiv preprint arXiv:2311.17043},
  year={2023}
}
```
11) [MiniGPT4-Video](https://github.com/Vision-CAIR/MiniGPT4-video/tree/main)
```bibtex
@article{ataallah2024minigpt4,
  title={Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens},
  author={Ataallah, Kirolos and Shen, Xiaoqian and Abdelrahman, Eslam and Sleiman, Essam and Zhu, Deyao and Ding, Jian and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2404.03413},
  year={2024}
}
```
12) [MovieChat](https://github.com/rese1f/MovieChat)
```bibtex
@inproceedings{song2024moviechat,
  title={Moviechat: From dense token to sparse memory for long video understanding},
  author={Song, Enxin and Chai, Wenhao and Wang, Guanhong and Zhang, Yucheng and Zhou, Haoyang and Wu, Feiyang and Chi, Haozhe and Guo, Xun and Ye, Tian and Zhang, Yanting and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18221--18232},
  year={2024}
}
```
13) [MA-LMM](https://github.com/boheumd/MA-LMM/tree/main)
```bibtex
@inproceedings{he2024malmm,
  title = {MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding},
  author    = {He, Bo and Li, Hengduo and Jang, Young Kyun and Jia, Menglin and Cao, Xuefei and Shah, Ashish and Shrivastava, Abhinav and Lim, Ser-Nam},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```

**[Ablation study]** 
1) [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)
2) [Vicuna-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5)
3) [Llama 2-7B](https://huggingface.co/meta-llama/Llama-2-7b)
4) [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)



## 📑 Citation
TODO
