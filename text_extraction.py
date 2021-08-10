from numpy.core.numeric import full
import pdf2image
import argparse
import PIL
import easyocr
import numpy as np
import os
import cv2
from openvino.inference_engine import IECore
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#argument

parser = argparse.ArgumentParser()
parser.add_argument('--file_path',type=str,dest='fp')
parser.add_argument('--dir_result',type=str,dest='dr')
parser.add_argument('--models_path',type=str,dest='mp')
args = parser.parse_args()

class Tools:

    def work_with_image(current_path):
        im_path=current_path
        d=0
        if args.fp[-3:]=='pdf':
            im=pdf2image.convert_from_path(args.fp)[0]
            im_path=im_path[-3:]+'jpg'
            im.save(im_path)
            d=1
        else:
            im=PIL.Image.open(args.fp)
        return im,im_path,d

    def convert_result_to_image(result):
        result = result.squeeze(0).transpose(1, 2, 0)
        result *= 255
        result[result < 0] = 0
        result[result > 255] = 255
        result = result.astype(np.uint8)
        return result
    
    def write_all(result,path):
        with open(path, 'w', encoding="utf-8") as f:
            for coordinates,txt in result:
                writing=''
                for c in coordinates:
                    writing+=str(c)
                    if c!=coordinates[-1]:
                        writing+=','
                writing+=' : '+txt 
                f.write(writing)
                f.write('\n')
            f.close()

class Model:
    def __init__(self,task):
        self.task=task
        if task=='reading text':
            self.model=pytesseract
        if task=='upscaling box':
            ie = IECore()
            net = ie.read_network(args.mp+'single-image-super-resolution-1032.xml', args.mp+'single-image-super-resolution-1032.bin')
            exec_net = ie.load_network(network=net, device_name="CPU")
            original_image_key = list(exec_net.input_info)[0]
            bicubic_image_key = list(exec_net.input_info)[1]
            output_key = list(exec_net.outputs.keys())[0]
            input_height, input_width = tuple(
            exec_net.input_info[original_image_key].tensor_desc.dims[2:])
            target_height, target_width = tuple(exec_net.input_info[bicubic_image_key].tensor_desc.dims[2:])
            self.model=original_image_key,bicubic_image_key,output_key,input_height,input_width,target_height,target_width,exec_net
        if task=='detecting text':
            self.model=easyocr.Reader(['pl'])

    def pred(self,input): 
        if self.task=='reading text':
            return self.model.image_to_string(input,lang='eng+pol')
        if self.task=='upscaling box':
            original_image_key,bicubic_image_key,output_key,input_height,input_width,target_height,target_width,exec_net=self.model
            w,h=input.size
            full_image=np.array(input)
            bicubic_image = cv2.resize(full_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            full_image = cv2.resize(full_image, (input_width, input_height))
            input_image_original = np.expand_dims(full_image.transpose(2, 0, 1), axis=0)
            input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)
            network_result = exec_net.infer(inputs={original_image_key: input_image_original,bicubic_image_key: input_image_bicubic,})
            result_image = Tools.convert_result_to_image(network_result[output_key])
            return PIL.Image.fromarray(result_image).resize((int(w*2.5),int(h*2.5))).convert('L')
        if self.task=='detecting text':
            return self.model.detect(input,min_size=1,low_text=0.3,link_threshold=0.1,text_threshold=0.1)[0]

class Box:
    def __init__(self,x_min,x_max,y_min,y_max,im,text=None):
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.im=im.crop((x_min,y_min,x_max,y_max))
        self.text=text

    def get_coordinates(self):
        return [self.x_min,self.x_max,self.y_min,self.y_max]
    
    def get_text(self):
        return self.text

    def right_distance2(self,box):
        return (self.x_max-box.x_min)**2+(self.y_min-box.y_min)**2
    
    def under_distance2(self,box):
        return (self.x_min-box.x_min)**2+(self.y_max-box.y_min)**2
    
    def equal(self,box):
        return self.x_min==box.x_min and self.x_max==box.x_max and self.y_min==box.y_min and self.y_max==box.y_max
    
    def set_text(self,txt):
        self.text=txt
    
    def upscale(self):
        upscaler=Model('upscaling box')
        self.im=upscaler.pred(self.im)

    def get_im(self):
        return self.im

class Boxes:
    def __init__(self):
        self.boxes_list=[]
        self.size=0
    
    def is_already_in(self,box):
        for bbox in self.boxes_list:
            if bbox.equal(box):
                return True
        return False
    
    def add_box(self,box):
        if not self.is_already_in(box):
            self.boxes_list.append(box)
            self.size+=1
    
    def set_texts(self):
        reader=Model('reading text')
        for i in range(self.size):
            text_predict=reader.pred(self.boxes_list[i].im)
            self.boxes_list[i].set_text(text_predict)
    
    def get_boxes_and_text(self):
        res=[]
        for i in range(self.size):
            box=self.boxes_list[i]
            res.append((box.get_coordinates(),box.get_text()))
        return res
     
class Image:
    def __init__(self,im,im_path):
        self.im=im 
        self.im_path=im_path
        self.boxes=Boxes()

    def get_text_boxes(self):
        text_detecter=Model('detecting text')
        text_aeras=text_detecter.pred(self.im_path)
        for (x_min,x_max,y_min,y_max) in text_aeras:
            self.boxes.add_box(Box(x_min,x_max,y_min,y_max,self.im))
    
    def upscale_boxes(self):
        for i in range(self.boxes.size):
            self.boxes.boxes_list[i].upscale()
    
    def get_features(self):
        return np.array(self.boxes.get_boxes_and_text(),dtype=object)
    
    def recognize_text(self):
        self.boxes.set_texts()
    
    def set_texts(self):
        self.get_text_boxes()
        self.upscale_boxes()
        self.recognize_text()

if __name__ == '__main__':

    im,im_path,d=Tools.work_with_image(args.fp)
    im=Image(im,im_path)

    im.set_texts()

    result=im.get_features()

    Tools.write_all(result,args.dr+'results.txt')
            
    if d==1:
        os.remove(im_path)
    
