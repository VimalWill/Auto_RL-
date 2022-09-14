import pdf2image
import PyPDF2
import numpy as np

pdf_file= '/content/prem.pdf'

import layoutparser as lp

model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})


obj_file = open(pdf_file,'rb')
pdf_reader =  PyPDF2.PdfFileReader(obj_file)

for i in range(pdf_reader)
    img = np.asarray(pdf2image.convert_from_path(pdf_file)[i])
    layout_result = model.detect(img)
    lp.draw_box(img, layout_result,  box_width=6, box_alpha=0.2, show_element_type=True)
    