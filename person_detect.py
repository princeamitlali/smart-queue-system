import os
import cv2
import argparse
import sys
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore


class Queue:

    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for queue in self.queues:
            x_minimum, y_minimum, x_maximum, y_maximum=queue
            frame=image[y_minimum:y_maximum, x_minimum:x_maximum]
            return frame
    
    def check_coords(self, coords, initial_w, initial_h): 
        data={key+1:0 for key in range(len(self.queues))}
        
        dummy = ['0', '1' , '2', '3']
        
        for coord in coords:
            x_minimum = int(coord[3] * initial_w)
            y_minimum = int(coord[4] * initial_h)
            x_maximum = int(coord[5] * initial_w)
            y_maximum = int(coord[6] * initial_h)
            
            dummy[0] = x_minimum
            dummy[1] = y_minimum
            dummy[2] = x_maximum
            dummy[3] = y_maximum
            
            for idx, queue in enumerate(self.queues):
                if dummy[0]>queue[0] and dummy[2]<queue[2]:
                    data[idx+1]+=1
        return data


class PersonDetect:
    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("enterred the correct model path")
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1) 
        # print('Network Loaded...')
        
    def predict(self, image):      
        inp_img = image

       
        num, channel, height, weight = self.input_shape
        image = cv2.resize(image, (weight, height))
        image = image.transpose((2, 0, 1))
        image = image.reshape((num, channel, height, weight))
        input_dict={self.input_name: image}  

        infer_request_handle = self.net.start_async(request_id=0, inputs=input_dict)
        infer_status = infer_request_handle.wait()
        if infer_status == 0:
            res = infer_request_handle.outputs[self.output_name]    
        return res, inp_img
        
    def draw_outputs(self, coords, frame, initial_w, initial_h):
        current_count = 0
        det = []        
        for objct in coords[0][0]:
            if objct[2] > self.threshold:
                x_minimum = int(objct[3] * initial_w)
                y_minimum = int(objct[4] * initial_h)
                x_maximum = int(objct[5] * initial_w)
                y_maximum = int(objct[6] * initial_h)
                cv2.rectangle(frame, (x_minimum, y_minimum), (x_maximum, y_maximum), (0, 55, 255), 1)
                current_count = current_count + 1
                det.append(objct)
                
        return frame, current_count, det


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    prsn_detct= PersonDetect(model, device, threshold)
    prsn_detct.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for queue in queue_param:
            queue.add_queue(queue)       
        print(queue_param)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    prsn_cnt=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            prsn_cnt+=1
            coords, image= prsn_detct.predict(frame)
            frame, current_count, coords = prsn_detct.draw_outputs(coords, image, initial_w, initial_h)
            #print(coords)
        
            num_of_people = queue.check_coords(coords, initial_w, initial_h)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_of_people}")
            
            text_out=""
            y_pixel=25
            
            for key, value in num_of_people.items():
                print(key, value)
                text_out += f"No. of People in Queue {key} is {value} "
                if value >= int(max_people):
                    text_out += f" Queue full; Please move to next Queue "
                cv2.putText(image, text_out, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                text_out=""
                y_pixel+=40

            out_video.write(image)
            
        total_time=time.time()-start_inference_time    
        ttl_inf_time=round(total_time, 1)
        fps=prsn_cnt/ttl_inf_time

        with open(os.path.join(output_path, 'stats.txt'), 'weight') as f:
            f.write(str(ttl_inf_time)+'\num')
            f.write(str(fps)+'\num')
            f.write(str(total_model_load_time)+'\num')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)