# PENDIENTES
 - [x] Guardar pesos del modelo durante entrenamiento
 - [ ] Agregar jitter en transforms y ver si mejora
 - [ ] **Post** procesado de mascaras por si quedan con ruido o hueco
 - [x] Nuevo umbral sigmoid
 - [ ] Probar learning rate scheduler
 - [ ] (!!) Arreglar funcion de submission para que funcione bien para batch_size > 1. Actualmente hay que correr el dataloader de kaggle con batch_size=1 antes de llamar a la funcion de la submission para que retorne bien la cantidad de filas.
 - [ ] Investigar efecto de preprocesado de imagenes, mas alla del data augmentation
 - [ ] FOCAL TVERSKY LOSS
 - [ ] Fine tunning
 - [ ] Scheduler learning rate
 - [ ] Investigar efecto de **pre**procesado de imagenes, mas alla del data augmentation
 - [ ] Tratar el problema como si tuviera clases desbalanceadas -- hay mas fondo que persona
 - [ ] Probar que sobre el final use la Focal Loss
 - [ ] Asegurarte que la máscara use Resize con NEAREST


# Mejoras unet:
- [ ] double convalucion
- 

 
 
# RESULTADOS

| Experimento (aprox)                                            | Celda | Modelo            | Dice (test) |
| -------------------------------------------------------------- | ----- | ----------------- | ----------- |
| UNet original, B/N, padding 0                                  | 75    | `model_1_2`       | **0.12**    |
| UNet, padding=1, B/N                                           | 80    | `unet_pad_1`      | **0.80**    |
| UNet, padding=1, B/N                                           | 80    | `unet_pad_1`      | **0.84**    |
| UNet, padding=1, RGB                                           | 87    | `unet_rgb`        | **0.83**    |
| UNet, RGB, BN + Dropout                                        | 94    | `unet_rgb_2`      | **0.89**    |
| UNet, RGB, solo Dropout                                        | 100   | `unet_rgb_drop`   | **0.82**    |
| UNet, RGB, BN + Dropout + data aug                             | 107   | `unet_rgb_3`      | **0.80**    |
| UNet, B/N, Dice + BCE                                          | 117   | `unet_dice`       | **0.68**    |
| UNet, RGB, Dice + BCE                                          | 121   | `unet_rgb_dice`   | **0.81**    |
| UNet, RGB, Dice + BCE + “nuevas transforms”                    | 129   | `unet_rgb_dice_2` | **0.71**    |
| UNetAttention, RGB, Dropout + DA, Dice + BCE                   | 136   | `unet_att`        | **0.82**    |
| UNetAttention re-entrenado (loss ponderada + LR un poco mayor) | 150   | `unet_att_2`      | **0.81**    |
| UNetAttention, RGB, BN, Dropout, SIN data aug                  | 150   | `unet_att_3`      | **0.88**    |
| UNetAttention, RGB, BN, Dropout, SIN data aug + POST PROC      | 150   | `unet_att_3`      | **0.90**    |
| UNetAttention, RGB, BN, Dropout + Crop                         | 150   | `unet_crop`       | **0.86**    |
| UNetAttention, RGB, BN, Dropout + Crop (p=0.3)                 | 150   | `unet_crop`       | **0.88**    |
| UNetAttention, RGB, BN, Dropout + Crop (p=0.3)                 | 150   | `unet_crop`       | **0.88**    |
| UNetAttention, RGB, BN, Dropout + convolucion doble            | 150   | `unet_dc`         | **0.91**    |






# INFO

### Guardar pesos del modelo

Se guardan en el train() -> pasar por parametro nombre de archivo que deje claro cual era la arquitectura

Para restaurar:

    model = UNet(n_class=1, padding=1).to(DEVICE) #crear modelo con misma arquitectura
    
    checkpoint = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

Para reanudar entrenamiento:

    model = UNet(n_class=1, padding=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    checkpoint = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

### Clases y funciones
class SegDataset: Para vincular imagen/mascara

def load_image: 
carga la imagen, la pasa a blanco y negro o RGB segun corresponda, y devuelve el tensor

class TestSegmentationDataset: 
como SegDataset pero solo para la carpeta TEST y NO busca mascaras (porque no hay)

def get_seg_dataloaders: 
 * toma el dataset completo con SegDataset
 * genera train_ds , val_ds, test_ds : a partir de SegDataset
 * genera test_ds_kaggle : a partir de TestSegmentationDataset
 * devuelve los data loaders

def center_crop: lo usa UNet

def model_segmentation_report: 
evalua el modelo + calcula y muestra metricas. Hay que tener cuidado con los tamaños, porque si no tenemos padding, la red devuelve imagenes mas chicas que la mascara y para poder compararlas tienen que coincidir.

### Fine-tuning con una loss “más suave”, Estrategia práctica:

* Entrenar como ahora (BCE+Dice).
* Guardar el mejor checkpoint.
* Cargarlo y entrenar solo 10–20 épocas más con Focal Tversky o con Dice puro, con LR más chico (ej. 1e-4 o 3e-5).

### Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3, verbose=True
)

### Links

UNet con arquitectura modificada:
* https://iopscience.iop.org/article/10.1088/1742-6596/1815/1/012018/pdf

UNet + attention: 
* https://arxiv.org/pdf/1804.03999
* https://www.kaggle.com/code/utkarshsaxenadn/person-segmentation-attention-unet ---> en esta muestran predicted mask vs true mask durante training!!!

UNet for human segmentation, con arquitectura original: 
* https://towardsdev.com/human-segmentation-using-u-net-with-source-code-easiest-way-f78be6e238f9

FOCAL TVERSKY LOSS
* https://arxiv.org/pdf/1810.07842, usar esta loss en los ultimos 10/20 epoch 
* https://www.igi-global.com/pdf.aspx?tid=315756&ptid=310168&ctid=4&oa=true&isxn=9781668479315


