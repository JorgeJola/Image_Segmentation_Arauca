from flask import Blueprint, Flask, render_template, redirect, request,flash, url_for,send_file
import tempfile
import rasterio
import os
import numpy as np
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import slic



main = Blueprint('main', __name__)

UPLOAD_FOLDER = tempfile.mkdtemp()
RESULT_FOLDER = tempfile.mkdtemp()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@main.route('/', methods=["GET", "POST"])
def image_segmentation():
    if request.method == "POST":
        if 'raster' not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files['raster']
        municipality = request.form.get("municipality")

        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file:
            # Guardar archivo subido
            processed_raster_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(processed_raster_path)
            segmented_raster_path = segment_raster(processed_raster_path, municipality)
            return redirect(url_for('main.download_file', filename=os.path.basename(segmented_raster_path)))
        
    return render_template('image_segmentation.html', success=False)



def segment_raster(input_path, municipality):
    
    print('Empezamos')
    with rasterio.open(input_path) as src:
        nbands = src.count
        width = src.width
        height = src.height
        band_data = []

        # Leer las bandas en bloques y apilar
        for i in range(1, nbands + 1):
            band = src.read(i, window=rasterio.windows.Window(0, 0, width, height))
            band_data.append(band)
        band_data = np.dstack(band_data)
        
        # Ajustar intensidades y realizar segmentación
        img = exposure.rescale_intensity(band_data)
        segments = slic(img, n_segments=10, compactness=0.03)
        
        # Guardar el resultado de la segmentación
        output_path = os.path.join(RESULT_FOLDER, f"Segmented_Raster_{municipality}.tif")

        with rasterio.open(input_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(segments.astype(rasterio.float32), 1)
    print('Terminamos segmentacion')
    return output_path
    
@main.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)