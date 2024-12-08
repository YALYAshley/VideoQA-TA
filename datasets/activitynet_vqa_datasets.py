import os
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor
from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset
from lavis.datasets.data_utils import load_video

def generate_labelinput(video_name):
    
    json_file_path = os.path.join('/Activitynet_Zero_Shot_QA/ioudected/iouframerevised', 'v_' + f"{video_name}.json")
    

    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        print(f"JSON file for video '{json_file_path}' not found in jsonmerge_dir")
        return ""

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Get all final labels and their corresponding trajectories
    final_labels = []
    trajectories_by_label = {}

    for key, value in data.items():
        final_label = value["final_label"]
        final_labels.append(final_label)
        trajectories = value["trajectory"]
        trajectories_by_label.setdefault(final_label, []).extend(trajectories)

    # Initialize the labelinput with the main description
    labelinput = "This video showcases various objects appearing in different frames. Below are the details of each frame along with the coordinates of the objects (formatted as [x1, y1, x2, y2], where (x1, y1) are the top-left corner and (x2, y2) are the bottom-right corner):\n"

    # Dictionary to keep track of which objects appear in which frames
    included_frames = {}

    # Iterate over all final labels
    for final_label in final_labels:
        # Sort trajectories for the current label by frame length and representativeness
        representative_trajectories = sorted(
            trajectories_by_label[final_label],
            key=lambda x: (len(x["frame"]), x.get("representativeness", 0)),
            reverse=True
        )[:]

        # Iterate over the selected representative trajectories
        for trajectory in representative_trajectories:
                frame = trajectory['frame']
                box = trajectory['box']
                if frame not in included_frames:
                    included_frames[frame] = []
                if not any(label == final_label for label, _ in included_frames[frame]):
                    included_frames[frame].append((final_label, box))

    frame_count = 0

    # Iterate over the frames in sorted order to build the labelinput
    for frame, objects in sorted(included_frames.items()):
            frame_count += 1
            # Create the description for the current frame
            frame_str = f"In the {frame_count} frame, it includes: "
            object_descriptions = [f"{obj} with {box}" for obj, box in objects]
            frame_str += ', '.join(object_descriptions) + ".\n"
            # Append the frame description to the prompt
            labelinput += frame_str

            # Check if labelinput length is nearing the limit
            if len(labelinput) >= 2000:  # Adjust this threshold as needed to ensure the labelinput stays under 2048 characters
                break

    return labelinput

class ActivityNetVQADataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt='', split='train'):
        self.vis_root = vis_root

        self.annotation = {}
        for ann_path in ann_paths:
            self.annotation.update(json.load(open(ann_path)))
        self.question_id_list = list(self.annotation.keys())
        self.question_id_list.sort()
        self.fps = 10

        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.prompt = prompt

    def __getitem__(self, index):
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        # Divide the range into num_frames segments and select a random index from each segment
        segment_list = np.linspace(0, ann['frame_length'], self.num_frames + 1, dtype=int)
        segment_start_list = segment_list[:-1]
        segment_end_list = segment_list[1:]
        selected_frame_index = []
        for start, end in zip(segment_start_list, segment_end_list):
            if start == end:
                selected_frame_index.append(start)
            else:
                selected_frame_index.append(np.random.randint(start, end))

        frame_list = []
        last_frame = None
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)

        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        labelinput = generate_labelinput(ann['video'])
        # print(f'labelinput:{labelinput}')
        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            "labelinput": labelinput,
        }
        
    def __len__(self):
        return len(self.question_id_list)

class ActivityNetVQAEvalDataset(ActivityNetVQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_frames, prompt, split='test')

    def __getitem__(self, index):
        question_id = self.question_id_list[index]
        ann = self.annotation[question_id]

        selected_frame_index = np.rint(np.linspace(0, ann['frame_length']-1, self.num_frames)).astype(int).tolist()
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(os.path.join(self.vis_root, ann['video'], "frame{:06d}.jpg".format(frame_index + 1))).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)

        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        labelinput = generate_labelinput(ann['video'])
        question = self.text_processor(ann["question"])
        if len(self.prompt) > 0:
            question = self.prompt.format(question)
        answer = self.text_processor(ann["answer"])

        return {
            "image": video,
            "text_input": question,
            "text_output": answer,
            "question_id": ann["question_id"],
            "labelinput": labelinput,
        }
