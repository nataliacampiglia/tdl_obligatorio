class SegDataset: Para vincular imagen/mascara

def load_image: carga la imagen, la pasa a blanco y negro o RGB segun corresponda, y devuelve el tensor

class TestSegmentationDataset: como SegDataset pero solo para la carpeta TEST y NO busca mascaras (porque no hay)

def get_seg_dataloaders: 
    * toma el dataset completo con SegDataset
    * genera train_ds , val_ds, test_ds : a partir de SegDataset
    * genera test_ds_kaggle : a partir de TestSegmentationDataset
    * devuelve los data loaders

transforms: por ahora solamente a la imagen -> normalize

def center_crop: lo usa UNet

class UNet: arquitectura de la red

def model_segmentation_report: evalua el modelo + calcula y muestra metricas. Hay que tener cuidado con los tamaños, porque si no tenemos padding, la red devuelve imagenes mas chicas que la mascara y para poder compararlas tienen que coincidir.