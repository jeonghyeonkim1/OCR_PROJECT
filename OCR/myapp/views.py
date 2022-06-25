from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from PIL import Image
import pytesseract

# Create your views here.



def ocr(request):
    context = {}
    if 'uploadfile' in request.FILES:
        uploadfile = request.FILES.get('uploadfile', '')

        if uploadfile != '':
            name_old = uploadfile.name

            fs = FileSystemStorage(location='static/source')
            imgname = fs.save(f'src-{name_old}', uploadfile)
            imgfile = Image.open(f'./static/source/{imgname}')
            resulttext = pytesseract.image_to_string(imgfile, lang='kor')

    context['imgname'] = imgname
    context['resulttext'] = resulttext.replace(" ", "")

    return render(request, 'coocr_upload.html', context)
