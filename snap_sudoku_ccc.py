import io
import os

from PIL import Image

import dialogs
import photos
import ui
from objc_util import ObjCClass, ns, nsurl

MODEL_FILENAME = "MNIST.mlmodel"
# MODEL_FILENAME = 'MNISTClassifier.mlmodel'
# MODEL_FILENAME = 'OCR.mlmodel'
# MODEL_FILENAME = 'Alphanum_28x28.mlmodel'

# Use a local path for caching the model file
MODEL_PATH = os.path.join(os.path.expanduser("~/Documents/"), MODEL_FILENAME)

# Declare/import ObjC classes:
MLModel = ObjCClass("MLModel")
VNCoreMLModel = ObjCClass("VNCoreMLModel")
VNCoreMLRequest = ObjCClass("VNCoreMLRequest")
VNImageRequestHandler = ObjCClass("VNImageRequestHandler")


def load_model() -> VNCoreMLModel:
    global vn_model
    # Compile the model:
    c_model_url = MLModel.compileModelAtURL_error_(nsurl(MODEL_PATH), None)
    # Load model from the compiled model file:
    ml_model = MLModel.modelWithContentsOfURL_error_(c_model_url, None)
    # Create a VNCoreMLModel from the MLModel for use with the Vision framework:
    vn_model = VNCoreMLModel.modelForMLModel_error_(ml_model, None)
    return vn_model


def pil2ui(pil_image: Image) -> ui.Image:
    with io.BytesIO() as bIO:
        pil_image.save(bIO, "PNG")
        return ui.Image.from_data(bIO.getvalue())


def _classify_img_data(img_data: ns) -> dict:
    global vn_model
    # Create and perform the recognition request:
    req = VNCoreMLRequest.alloc().initWithModel_(vn_model).autorelease()
    handler = (
        VNImageRequestHandler.alloc()
        .initWithData_options_(img_data, None)
        .autorelease()
    )
    success = handler.performRequests_error_([req], None)
    if not success:
        return None    
    best_result = req.results()[0]
    return {"label": str(best_result.identifier()),
            "confidence": best_result.confidence()}


def classify_image(img):
    with io.BytesIO() as buffer:
        img.save(buffer, "JPEG")
        return _classify_img_data(ns(buffer.getvalue()))


def classify_asset(asset):
    mv = ui.View(bg_color="white")
    im = ui.ImageView()
    pil_image = asset.get_image()
    print(pil_image.size)
    ui_image = asset.get_ui_image()
    n_squares = 9
    d_grid = 15  # % around the digit
    wim, him = pil_image.size
    ws, hs = ui.get_screen_size()
    if (ws / hs) < (wim / him):
        h = ws * him / wim
        im.frame = (0, (hs - h) / 2, ws, h)
    else:
        w = hs * wim / him
        im.frame = ((ws - w) / 2, 0, w, hs)
    print(wim, him, ws, hs)
    mv.add_subview(im)
    wi = im.width
    hi = im.height
    im.image = ui_image
    im.content_mode = 1  # 1
    mv.frame = (0, 0, ws, hs)
    mv.present("fullscreen")
    dx = wim / n_squares
    dy = him / n_squares
    d = dx * d_grid / 100
    dl = int((wi / n_squares) * d_grid / 100)
    for ix in range(n_squares):
        x = ix * dx
        for iy in range(n_squares):
            y = iy * dy
            pil_char = pil_image.crop(
                (int(x + d), int(y + d), int(x + dx - d), int(y + dy - d))
            )
            l = ui.Button()
            l.frame = (
                int(ix * wi / n_squares) + dl,
                int(iy * hi / n_squares) + dl,
                int(wi / n_squares) - 2 * dl,
                int(hi / n_squares) - 2 * dl,
            )
            l.border_width = 1
            l.border_color = "red"
            l.tint_color = "red"
            ObjCInstance(l).button().contentHorizontalAlignment = 1  # left
            l.background_image = pil2ui(pil_char)
            im.add_subview(l)
            l.title = classify_image(pil_char)["label"]


def main():
    global vn_model
    vn_model = load_model()
    asset = photos.pick_asset(assets=photos.get_assets())
    if asset:
        classify_asset(asset)


if __name__ == "__main__":
    main()
