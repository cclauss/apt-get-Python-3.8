#!python3
'''
This is a demo of how you can use the CoreML framework (via objc_util) to classify images in Pythonista.
It downloads the trained 'MobileNet' CoreML model from the Internet, and uses it to classify images that
are either taken with the camera, or picked from the photo library.
'''

import requests
import os
import io
import photos
import dialogs
from PIL import Image
from objc_util import ObjCClass, nsurl, ns
import ui

# Configuration (change URL and filename if you want to use a different model):
MODEL_URL = 'https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel'
#MODEL_FILENAME = 'Alphanum_28x28.mlmodel'
MODEL_FILENAME = 'OCR.mlmodel'
#MODEL_FILENAME = 'frozen_east_text_detection.pb.py'

# Use a local path for caching the model file (no need to sync this with iCloud):
MODEL_PATH = os.path.join(os.path.expanduser('~/Documents'), MODEL_FILENAME)

# Declare/import ObjC classes:
MLModel = ObjCClass('MLModel')
VNCoreMLModel = ObjCClass('VNCoreMLModel')
VNCoreMLRequest = ObjCClass('VNCoreMLRequest')

VNImageRequestHandler = ObjCClass('VNImageRequestHandler')
VNDetectTextRectanglesRequest = ObjCClass('VNDetectTextRectanglesRequest')


def load_model():
    '''Helper method for downloading/caching the mlmodel file'''
    if not os.path.exists(MODEL_PATH):
        print(f'Downloading model: {MODEL_FILENAME}...')
        r = requests.get(MODEL_URL, stream=True)
        file_size = int(r.headers['content-length'])
        with open(MODEL_PATH, 'wb') as f:
            bytes_written = 0
            for chunk in r.iter_content(1024*100):
                f.write(chunk)
                print(f'{bytes_written/file_size*100:.2f}% downloaded')
                bytes_written += len(chunk)
        print('Download finished')
    ml_model_url = nsurl(MODEL_PATH)
    # Compile the model:
    c_model_url = MLModel.compileModelAtURL_error_(ml_model_url, None)
    # Load model from the compiled model file:
    ml_model = MLModel.modelWithContentsOfURL_error_(c_model_url, None)
    # Create a VNCoreMLModel from the MLModel for use with the Vision framework:
    vn_model = VNCoreMLModel.modelForMLModel_error_(ml_model, None)
    return vn_model


def _classify_img_data(img_data):
    '''The main image classification method, used by `classify_image` (for camera images) and `classify_asset` (for photo library assets).'''
    vn_model = load_model()
    # Create and perform the recognition request:
    req = VNCoreMLRequest.alloc().initWithModel_(vn_model).autorelease()
    handler = VNImageRequestHandler.alloc().initWithData_options_(img_data, None).autorelease()
    success = handler.performRequests_error_([req], None)
    if success:
        best_result = req.results()[0]
        label = str(best_result.identifier())
        confidence = best_result.confidence()
        return {'label': label, 'confidence': confidence}
    else:
        return None


def classify_image(img):
    buffer = io.BytesIO()
    img.save(buffer, 'JPEG')
    img_data = ns(buffer.getvalue())
    return _classify_img_data(img_data)
    
def classify_asset(asset):
    img_data = ns(asset.get_image_data().getvalue())
        
    req = VNDetectTextRectanglesRequest.alloc().init()
    req.reportCharacterBoxes = True
    
    handler = VNImageRequestHandler.alloc().initWithData_options_(img_data, None).autorelease()
    success = handler.performRequests_error_([req], None)
    if success:
        im = ui.ImageView()
        pil_image = asset.get_image()
        print(pil_image.size)
        ui_image = asset.get_ui_image()
        wim,him = ui_image.size
        im.frame = (0,0,400,400*him/wim)
        #im.frame = (0,0,141,64)
        wi = im.width
        hi = im.height
        im.image = ui_image
        im.content_mode = 1 #1
        im.present()
        for i in range(0,len(req.results())):
            observation = req.results()[i]  
            box = observation.boundingBox()
            xb=box.origin.x
            yb=box.origin.y
            wb=box.size.width
            hb=box.size.height
            #print('x=',xb)
            #print('y=',y )
            #print('width=',w )
            #print('height=',hb)
            l = ui.Label()
            l.frame = (xb*wi,yb*hi,wb*wi,hb*hi)
            #print(l.frame)
            #l.border_width = 1
            #l.border_color = 'red'
            im.add_subview(l)
            #print(dir(observation))
            confidence = observation.confidence()
            #print('confidence', confidence)
            for i_ch in range(0,len(observation.characterBoxes())):
              ch_box = observation.characterBoxes()[i_ch]
              box = ch_box.boundingBox()
              x=box.origin.x
              y=box.origin.y
              w=box.size.width
              h=box.size.height
              #print('x=',x)
              #print('y=',y)
              #print('width=',w)
              #print('height=',h)
              l = ui.Label()
              l.frame = (x*wi,yb*hi,w*wi,hb*hi)
              #print(l.frame)
              l.border_width = 1
              l.border_color = 'blue'
              im.add_subview(l)
              print((int(x*wim),int(yb*him),int(w*wim),int(hb*him)))
              pil_char = pil_image.crop((int(x*wim)-1,int(yb*him)-1,int((x+w)*wim)+1,int((yb+hb)*him)+8))
              pil_char.show()
              print(classify_image(pil_char))
              #print(dir(ch_box))
              #break
        print('ok')
    else:
        print('error')
        


def main():
    all_assets = photos.get_assets()
    asset = photos.pick_asset(assets=all_assets)
    if asset is None:
        return
    classify_asset(asset)

if __name__ == '__main__':
    main()
