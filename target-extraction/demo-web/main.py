from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pdf2image
import numpy as np
import json
import shutil
from openvino.inference_engine import IECore
import pytesseract

from layoutlm_preprocess import *

from pydantic import BaseModel, Json

mp='C:/Users/emili/Desktop/VivaDrive/models/trained_layoutlm.pt'

class Prediction(BaseModel):
    """
    Prediction Json Response Model
    """
    filename: str
    result: Json

app = FastAPI()

app.mount('/static', StaticFiles(directory='static/'))

templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):

    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/upload_pdf/",response_model=Prediction)
async def upload_pdf(request: Request, file: UploadFile = File(...)):

    if file.filename.endswith(".pdf"):

        try:
            with open('static/{}'.format(file.filename), "wb") as buffer:
                shutil.copyfileobj(file.file,buffer)
            labels=['S-DATE_ID','S-PLATE_ID','S-INVOICE_ID','S-CLIENT_ID','S-COSTTYPE_ID','S-DISTANCE_ID','S-QUANTITY_ID','S-PRICENETTO_ID','S-PRICEBUTTO_ID',
                        'S-SELLE_ID','S-BUYER_ID','O']
            im_list, im_path_list=Tools.work_with_image('static/{}'.format(file.filename))
            invoice=Image(im_list, im_path_list, labels)
            result=invoice.text_extraction(mp, 'static/')
            return {"filename": file.filename, "result": result}
            
        except Exception as e:
            #print(e)
            raise HTTPException(status_code=500,detail=str(e))

    else:
        raise HTTPException(status_code=500, detail="Not a .pdf file")

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class Tools:

    def label_map(labels):
        return {i: label for i, label in enumerate(labels)}

    def write_all_json(d,path):
        jfile=json.dumps(d)
        if len(path)!=0:
            jsonFile = open(path+'result.json', "w")
            jsonFile.write(jfile)
            jsonFile.close()
        return jfile

    def work_with_image(current_path):
        im_path=current_path
        if im_path[-3:]!='pdf':
            print('Please enter a pdf file')
        else:
            ims=pdf2image.convert_from_path(im_path)
            number_page=0
            im_list=[]
            im_path_list=[]
            for im in ims:
                im=im.resize((826,1169))
                im_list.append(im)
                im_path=im_path[:-4]+str(number_page)+'.jpg'
                im.save(im_path)
                im_path_list.append(im_path)
                number_page+=1
        return im_list, im_path_list  

class Model:

    def __init__(self, model_path, num_labels):
        self.model=model_load(model_path,num_labels)
    
    def pred(self, image, words, boxes, actual_boxes):
        word_level_predictions, predicted_boxes=convert_to_features(image, words, boxes, actual_boxes, self.model)
        return word_level_predictions, predicted_boxes

class Box:
    def __init__(self,x_min,y_min,x_max,y_max,text=None):
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.text=text

    def get_coordinates(self):
        return np.array([self.x_min, self.y_min, self.x_max, self.y_max])
    
    def equal(self,box):
        return self.x_min==box.x_min and self.x_max==box.x_max and self.y_min==box.y_min and self.y_max==box.y_max

    def distance2(self,box):
        return np.sum((self.get_coordinates()-box.get_coordinates())**2)

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
    
    def is_the_closest(self,box):
        ref=np.infty
        for i in range(self.size):
            d=box.distance2(self.boxes_list[i])
            if d<ref:
                ref=d
                res=self.boxes_list[i]
        return res
    
    def get_boxes_and_text(self):
        res=[]
        for i in range(self.size):
            box=self.boxes_list[i]
            res.append((box.get_coordinates(),box.get_text()))
        return res
     
class Image:

    def __init__(self, im_list, im_path_list, labels):
        self.im_path_list=im_path_list
        self.number_page=len(im_path_list)
        self.im_list=im_list
        self.image_list=[]
        self.words_list=[]
        self.boxes_list=[]
        self.actual_boxes_list=[]
        self.labels=labels
        self.boxes_detected_list=[Boxes() for i in range(self.number_page)]

    def set_texts(self):
        for i in range(self.number_page):
            image, words, boxes, actual_boxes = preprocess(self.im_path_list[i])
            self.image_list.append(image)
            self.words_list.append(words)
            self.boxes_list.append(boxes)
            self.actual_boxes_list.append(actual_boxes)
            for j in range(len(words)):
                box=Box(actual_boxes[j][0], actual_boxes[j][1], actual_boxes[j][2], actual_boxes[j][3], words[j])
                self.boxes_detected_list[i].add_box(box)
    
    def get_target_texts(self, model_path, path):
        d={}
        label_map=Tools.label_map(self.labels)
        model=Model(model_path, len(self.labels))
        for i in range(self.number_page):
            prediction_labels, prediction_boxes=model.pred(image=self.image_list[i], words=self.words_list[i], boxes=self.boxes_list[i],
                                                                         actual_boxes=self.actual_boxes_list[i])
            for prediction,box in zip(prediction_labels, prediction_boxes):
                predicted_label = label_map[prediction]
                if predicted_label!='O':
                    bbox=Box(box[0],box[1],box[2],box[3])
                    corresponding_box=self.boxes_detected_list[i].is_the_closest(bbox)
                    if predicted_label not in d:
                        d[predicted_label]=[corresponding_box.text]
                    else:
                        if corresponding_box.text not in d[predicted_label]:
                            d[predicted_label].append(corresponding_box.text)
        return Tools.write_all_json(d, path)  

    def text_extraction(self, model_path, result_path):
        self.set_texts()  
        return self.get_target_texts(model_path, result_path)    


